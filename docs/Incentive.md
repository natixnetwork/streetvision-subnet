# Natix Network Incentive Mechanism

This document covers the current state of the Natix Network's incentive mechanism, designed to encourage high-quality performance and continuous improvement among miners and validators.
This document covers the current state of SN34's incentive mechanism.
1. [Overview](#overview)
2. [Rewards for Miners](#Rewards for Miners)
2. [Rewards for Validators](#Rewards for Validators)
3. [Ranking and Incentives](#Ranking and Incentives)
5. [Incentive](#incentives)

## Overview

The Natix Network employs a dynamic reward system to incentivize miners to continuously improve their models for detecting construction site elements in images. Validators play a crucial role in maintaining the integrity and accuracy of the network.

## Rewards for Miners
Performance on video and image challenges are computed separately -- each is a weighted combination of the MCC of the last 100 predictions and the accuracy of the last 10.

Miners are rewarded based on their performance in detecting construction site elements. Their success rate, which determines their rank, is assessed by validators through a mix of organic tasks and tasks with known outcomes. This process ensures that miner models are accurately evaluated for their task performance.

- **Model Submission and Reward Period:**
  - Miners must submit at least one model to a publicly accessible repository, such as Hugging Face.
  - For the first 45 days after a model is submitted, miners receive the full reward for task performance.
  - After 45 days, the reward gradually decreases towards zero, encouraging miners to submit enhanced models.

- **Model Improvement and Evaluation:**
  - Miners can submit a new model at any time to reset the reward period for another 90 days.
  - The new model is evaluated against a set of known tests to ensure it performs better than the previous version.

## Rewards for Validators

Validators are rewarded for their role in safeguarding the network by assessing the accuracy of miners' work. They ensure fairness and precision in ranking miners, which directly influences the distribution of rewards.

## Ranking and Incentives

- **Ranking System:**
  - Validators rank miners based on the accuracy of their model predictions in mixed task scenarios.
  - The rank assigned by validators determines the distribution of rewards among miners, incentivizing high-quality predictions and consistent performance.

## Incentives

The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) is used to translate the rank and performance data into incentives for subnet miners and dividends for validators. This mechanism ensures that rewards are fairly distributed based on performance metrics, encouraging continued participation and model refinement.

By maintaining a focus on model improvement and task accuracy, the Natix Network aims to foster a robust and efficient system for detecting construction site elements, supporting both innovation and reliability within the network.
