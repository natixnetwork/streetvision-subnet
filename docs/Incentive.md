# Network Incentive Mechanism

This document covers the current state of the network's incentive mechanism, designed to encourage high-quality performance and continuous improvement among miners and validators.
This document covers the current state of SN72's incentive mechanism.
1. [Overview](#overview)
2. [Rewards for Miners](#Rewards-for-Miners)
2. [Rewards for Validators](#Rewards-for-Validators)
3. [Ranking and Incentives](#Ranking-and-Incentives)
5. [Incentive](#incentives)

## Overview

The network employs a dynamic reward system to incentivize miners to continuously improve their models for detecting construction site elements in images. Validators play a crucial role in maintaining the integrity and accuracy of the network.

## Rewards for Miners
Miners are rewarded based on their performance in classfiying construction sites. Their success rate, which determines their rank, is assessed by validators through a mix of organic tasks and tasks with known outcomes. This process ensures that miner models are accurately evaluated for their task performance.

- **Initial Model Submission – Current Requirement:**
  - Miners must <a href="Mining.md#Submitted-a-model">submit at least one model</a> to a publicly accessible repository on Hugging Face.
  - Any models that classfiying construction sites is accpted

- **Regular Model Submission and Reward Period – Future Plan:**
  - Miners must <a href="Mining.md#Submitted-a-model">submit an improved model on a regular basis</a>.
  - For the first 45 days after a model is submitted, miners receive the full reward for tasks performed.
  - After 45 days, the reward gradually decreases toward zero, encouraging miners to submit enhanced models.

- **Model Improvement and Evaluation - Future Plan:**
  - Miners can submit a new model at any time to reset the reward period for another 90 days.
  - The new model is evaluated against a known test set (publicly accessible repository on Hugging Face).
  - Only models that perform better than any previously submitted model are accepted.

**Note: The 90-day reward period and the 45-day split are subject to change. More details will be published to the community in advance before the Regular Model Submission requirements take effect.**
 
## Rewards for Validators

Validators are rewarded for their role in safeguarding the network by assessing the accuracy of miners' work. They ensure fairness and precision in ranking miners, which directly influences the distribution of rewards.

## Ranking and Incentives

- **Ranking System:**
  - Validators rank miners based on the accuracy of their model predictions in mixed task scenarios.
  - The rank assigned by validators determines the distribution of rewards among miners, incentivizing high-quality predictions and consistent performance.

## Incentives

The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) is used to translate the rank and performance data into incentives for subnet miners and dividends for validators. This mechanism ensures that rewards are fairly distributed based on performance metrics, encouraging continued participation and model refinement.

By maintaining a focus on model improvement and task accuracy, the network aims to foster a robust and efficient system for detecting construction site elements, supporting both innovation and reliability within the network.
