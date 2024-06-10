#!/bin/bash

# Ensure enough space
rm -rd models/*

bash scripts/train/multitask/train_strength_detection_all.sh ${1:-42}
rm -rd models/*
bash scripts/train/multitask/train_strength_evidence_all.sh ${1:-42}
rm -rd models/*
bash scripts/train/multitask/train_detection_evidence_all.sh ${1:-42}
rm -rd models/*
bash scripts/train/multitask/train_strength_detection_evidence_all.sh ${1:-42}
rm -rd models/*
