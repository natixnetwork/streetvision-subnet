# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import asyncio
import copy
import os
import threading
import time
from traceback import print_exception
from typing import List, Union

import bittensor as bt
import joblib
import numpy as np

from natix.base.neuron import BaseNeuron
from natix.base.utils.weight_utils import convert_weights_and_uids_for_emit, process_weights_for_netuid
from natix.utils.config import add_validator_args
from natix.utils.mock import MockDendrite
from natix.validator.miner_performance_tracker import MinerPerformanceTracker

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        self.performance_trackers = {
            "image": None,
        }

        self.image_history_cache_path = os.path.join(self.config.neuron.full_path, "image_miner_performance_tracker.pkl")
        self.load_miner_history()

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        # Initialize as None, will be created lazily in async context
        self._dendrite = None
        bt.logging.info("Dendrite will be initialized on first use")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        # Initialize scores - will be overridden by load_state() if state file exists
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        # Load state before sync to preserve scores from previous runs
        self.load_state()

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    (
                        f"Running validator {self.axon} ",
                        f"on network: {self.config.subtensor.chain_endpoint} ",
                        f"with netuid: {self.config.netuid}",
                    )
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    @property
    def dendrite(self):
        """Lazy initialization of dendrite in async context"""
        if self._dendrite is None:
            if self.config.mock:
                from natix.utils.mock import MockDendrite
                self._dendrite = MockDendrite(wallet=self.wallet)
            else:
                self._dendrite = bt.dendrite(wallet=self.wallet)
            bt.logging.info(f"Dendrite initialized: {self._dendrite}")
        return self._dendrite

    async def concurrent_forward(self):
        coroutines = [self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network.
        The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step.
        The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        
        # Restart loop - handles full cycle restarts on critical errors
        while True:
            try:
                self._run_main_loop()
                break  # Normal exit
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()
            except Exception as err:
                bt.logging.error(f"CRITICAL ERROR in validator main loop: {str(err)}")
                bt.logging.error(f"Error type: {type(err).__name__}")
                bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))
                bt.logging.warning("=" * 60)
                bt.logging.warning("VALIDATOR RESTART TRIGGERED - Sleeping 60 seconds before restart")
                bt.logging.warning("=" * 60)
                # Sleep and restart the entire cycle
                time.sleep(60)
                bt.logging.info("RESTARTING validator main loop after critical error")
                # Loop continues to restart

    def _run_main_loop(self):
        """Main validator loop that can be restarted cleanly"""
        
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info("=" * 60)
        bt.logging.info(f"VALIDATOR MAIN LOOP STARTED - Block: {self.block}, Step: {self.step}")
        bt.logging.info("=" * 60)
        
        # Track health metrics
        last_successful_forward = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5

        # This loop maintains the validator's operations until intentionally stopped.
        while True:
            bt.logging.info(f"step({self.step}) block({self.block})")

            if self.config.proxy.port:
                try:
                    self.validator_proxy.get_credentials()
                    bt.logging.info("Validator proxy ping to proxy-client successfully")
                except Exception as e:
                    bt.logging.warning(e)
                    bt.logging.warning("Warning, proxy can't ping to proxy-client.")

            # Run multiple forwards concurrently with error handling
            try:
                self.loop.run_until_complete(self.concurrent_forward())
                last_successful_forward = time.time()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                bt.logging.error(f"Error in concurrent_forward ({consecutive_errors}/{max_consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    bt.logging.error("=" * 60)
                    bt.logging.error(f"CONSECUTIVE ERROR LIMIT REACHED ({consecutive_errors}/{max_consecutive_errors})")
                    bt.logging.error("VALIDATOR RESTART TRIGGERED - Too many consecutive forward errors")
                    bt.logging.error("=" * 60)
                    return  # Return to trigger full restart

            # Check if we should exit.
            if self.should_exit:
                break

            # Check health status
            time_since_last_forward = time.time() - last_successful_forward
            if time_since_last_forward > 600:  # 10 minutes
                bt.logging.warning(f"No successful forward in {time_since_last_forward:.0f} seconds")
            
            # Sync metagraph and potentially set weights.
            try:
                self.sync()
            except Exception as e:
                bt.logging.error(f"Error during sync: {e}")
                # Continue running even if sync fails
            
            time.sleep(60)
            self.step += 1

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners.
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                "Scores contain NaN values. "
                "This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(uids=processed_weight_uids, weights=processed_weights)
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph with timeout to prevent hanging
        try:
            # Run sync in a separate thread with timeout since it's a blocking call
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.metagraph.sync, subtensor=self.subtensor)
                future.result(timeout=60)  # 60-second timeout
        except concurrent.futures.TimeoutError:
            bt.logging.error("Metagraph sync timed out after 60 seconds")
            return
        except Exception as e:
            bt.logging.error(f"Error during metagraph sync: {e}")
            return

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning("Either rewards or uids_array is empty. No updates will be performed.")
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = self.scores.copy()
        vali_uids = [
            uid
            for uid in range(len(scattered_rewards))
            if self.metagraph.validator_permit[uid] and self.metagraph.S[uid] > self.config.neuron.vpermit_tao_limit
        ]
        scattered_rewards[vali_uids] = 0.0
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = alpha * scattered_rewards + (1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_miner_history(self):
        bt.logging.info(f"Saving miner performance history to {self.image_history_cache_path}")
        joblib.dump(self.performance_trackers["image"], self.image_history_cache_path)

    def load_miner_history(self):
        def load(path):
            if os.path.exists(path):
                bt.logging.info(f"Loading miner performance history from {path}")
                try:
                    tracker = joblib.load(path)
                    num_miners_history = len(
                        [
                            uid
                            for uid in tracker.prediction_history
                            if len([p for p in tracker.prediction_history[uid] if p != -1]) > 0
                        ]
                    )
                    bt.logging.info(f"Loaded history for {num_miners_history} miners")
                except Exception as e:
                    bt.logging.error(f"Error loading miner performance tracker: {e}")
                    tracker = MinerPerformanceTracker()
            else:
                bt.logging.info(f"No miner performance history found at {path} - starting fresh!")
                tracker = MinerPerformanceTracker()
            return tracker

        try:
            self.performance_trackers["image"] = load(self.image_history_cache_path)
        except Exception:
            # just for 2.0.0 upgrade for miner performance to carry over
            v1_history_cache_path = os.path.join(self.config.neuron.full_path, "miner_performance_tracker.pkl")
            self.performance_trackers["image"] = load(v1_history_cache_path)

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            os.path.join(self.config.neuron.full_path, "state.npz"),
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )
        self.save_miner_history()

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state_path = os.path.join(self.config.neuron.full_path, "state.npz")
        # Load the state of the validator from file.
        if os.path.exists(state_path):
            state = np.load(state_path)
            self.step = state["step"]
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
        else:
            bt.logging.warning(f"Warning: no state file available at {state_path}")
        self.load_miner_history()
