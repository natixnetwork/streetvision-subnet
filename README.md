<p align="center">
</p>
<h1 align="center">Natix Subnet<br><small>Bittensor Subnet XXX | Global Smart Maps</small></h1>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The Natix Subnet is designed to be the backbone of the first decentralized smart map. With Natixs world-class dashcam data pipeline and Bittensors DeAI infrastructure, we are building the worlds most comprehensive and dynamic smart map.

<table style="border: none !important; width: 100% !important; border-collapse: collapse !important; margin: 0 auto !important;">
  <tbody>
    <tr>
      <td><b>Docs</b></td>
      <td><b>Resources</b></td>
    </tr>
    <tr style="vertical-align: top !important">
      <td>
        ⛏️ <a href="docs/Mining.md">Mining Guide</a><br>
        🔧 <a href="docs/Validating.md">Validator Guide</a><br>
        🏗️ <a href="#Subnet-Architecture">Architecture Diagrams</a><br>
        📈 <a href="docs/Incentive.md">Incentive Mechanism</a><br>
        🤝 <a href="docs/Contributor_Guide.md">Contributor Guide</a></td>
      <td>
        🚀 <a href="https://www.bitmind.ai/apps">TODO: FRONTEND LINK</a><br>
        🤗 <a href="https://huggingface.co/natix-network-org">Natix Huggingface</a><br>
        📊 <a href="https://wandb.ai/bitmindai/bitmind-subnet">TODO Mainnet XX W&B</a> | <a href="https://wandb.ai/bitmindai/bitmind">Testnet 323 W&B</a><br>
        📖 <a href="docs/Glossary.md">Project Structure and Terminology</a><br>
        <a href="https://docs.bittensor.com/learn/bittensor-building-blocks">🧠 Bittensor Introduction</a><br> 
      </td>
    </tr>
  </tbody>
</table>


## Decentralized Detection of Driving Qualia
Who hasn't been late to work for reasons outside of your control? Roadwork, lane closures, accidents, weather, and other unforeseen circumstances can cause any number of annoyances. Centralized map providers cannot be everywhere at once, and they 


## Core Components

> This documentation assumes basic familiarity with Bittensor concepts. For an introduction, please check out the docs: https://docs.bittensor.com/learn/bittensor-building-blocks. 

**Miners** 
- Miners are tasked with running binary classifiers that discern between images with and without roadwork, and are rewarded based on their accuracy. 
- Miners predict a float value in [0., 1.], with values greater than 0.5 indicating the image contains roadwork.


**Validators** 
- Validators challenge miners with a balanced mix of real and synthetic media drawn from a diverse pool of sources.
- We continually add new datasets and generative models to our validators in order to maximize coverage of the types of diverse data. Models and datasets are defined in  `natix/validator/config.py`.


## Subnet Architecture


![Subnet Architecture](static/Subnet-Arch.png)

<details>
<summary align=center><i>Figure 1 (above): Ecosystem Overview</i></summary>
<br>

> This diagram provides an overview of the validator neuron, miner neuron, and other components external to the subnet.

- The green arrows show how applications interact with the subnet to provide AI-generated image and video detection functionality.
- The blue arrows show how validators generate synthetic data, challenge miners and score their responses.

</details>

<br>


![Subnet Architecture](static/Vali-Arch.png)

<details>
<summary align=center><i>Figure 2 (above): Validator Components</i></summary>
<br>

> This diagram presents the same architecture as figure 1, but with organic traffic ommitted and with a more detailed look at scored challenges and the associated validator neuron components.


**Challenge Generation and Scoring (Blue Arrows)**

For each challenge, the validator randomly samples a real or synthetic image/video from the cache, applies random augmentations to the sampled media, and distributes the augmented data to 50 randomly selected miners for classification. It then scores the miners responses, and logs comprehensive challenge results to [Weights and Biases](https://wandb.ai/bitmindai/bitmind-subnet), including the generated media, original prompt, miner responses and rewards, and other challenge metadata.

**Synthetic Data Generation (Pink Arrows)**:

The synthetic data generator coordinates a VLM and LLM to generate prompts for our suite of text-to-image, image-to-image, and text-to-video models. Each image or video is written to the cache along with the prompt, generation parameters, and other metadata.

**Dataset Downloads (Green Arrows)**:

The real data fetcher performs partial dataset downloads, fetching random compressed chunks of datasets from HuggingFace and unpacking random portions of these chunks into the cache along with their metadata. Partial downloads avoid requiring TBs of space for large video datasets like OpenVid1M.

</details>



## Community

<p align="left">
  <a href="https://discord.gg/kKQR98CrUn">
    <img src="static/Join-BitMind-Discord.png" alt="Join us on Discord" width="60%">
  </a>
</p>

For real-time discussions, community support, and regular updates, <a href="https://discord.gg/kKQR98CrUn">join our Discord server</a>. Connect with developers, researchers, and users to get the most out of Natix Subnet.

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
```
