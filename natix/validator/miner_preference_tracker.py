import time
from typing import Dict, List

import bittensor as bt


class MinerPreferenceTracker:
    """
    Tracks miner challenge preferences with persistence.
    """

    def __init__(self):
        self.preferences: Dict[int, List[int]] = {}
        self.miner_hotkeys: Dict[int, str] = {}
        self.last_updated: Dict[int, float] = {}

    def update_preferences(self, uid: int, preferences: List[int], miner_hotkey: str):
        """
        Update the preferences for a miner.
        """
        if uid not in self.preferences or self.miner_hotkeys.get(uid) != miner_hotkey:
            self.reset_miner_preferences(uid, miner_hotkey)

        self.preferences[uid] = preferences.copy()
        self.last_updated[uid] = time.time()
        
        # Convert IDs to names for human-readable logging
        from natix.validator.config import CHALLENGE_TYPE
        preference_names = [CHALLENGE_TYPE.get(pref_id, f"Unknown({pref_id})") for pref_id in preferences]
        bt.logging.debug(f"Updated preferences for miner {uid}: {preference_names} (IDs: {preferences})")

    def reset_miner_preferences(self, uid: int, miner_hotkey: str):
        """
        Reset the preferences for a miner.
        """
        self.preferences[uid] = []
        self.miner_hotkeys[uid] = miner_hotkey
        self.last_updated[uid] = time.time()

    def get_preferences(self, uid: int) -> List[int]:
        """
        Get the preferences for a miner.
        """
        return self.preferences.get(uid, [])

    def has_preference(self, uid: int, challenge_type: int) -> bool:
        """
        Check if a miner has a specific challenge type preference.
        """
        preferences = self.get_preferences(uid)
        return challenge_type in preferences

    def get_miners_with_preference(self, challenge_type: int) -> List[int]:
        """
        Get all miner UIDs that prefer a specific challenge type.
        """
        return [uid for uid, prefs in self.preferences.items() if challenge_type in prefs]

    def get_all_preferences(self) -> Dict[int, Dict]:
        """
        Get all miner preferences with metadata for reporting.
        """
        result = {}
        for uid in self.preferences:
            result[uid] = {
                'preferred_challenges': self.preferences[uid],
                'miner_hotkey': self.miner_hotkeys.get(uid, ''),
                'last_updated': self.last_updated.get(uid, 0)
            }
        return result

    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """
        Remove entries that haven't been updated recently.
        """
        current_time = time.time()
        old_uids = [
            uid for uid, last_update in self.last_updated.items()
            if current_time - last_update > max_age_seconds
        ]
        
        for uid in old_uids:
            self.preferences.pop(uid, None)
            self.miner_hotkeys.pop(uid, None)
            self.last_updated.pop(uid, None)
            bt.logging.debug(f"Cleaned up old preference entry for miner {uid}")