#!/bin/bash

## abortion
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_abortion_test.csv -motion-cross_topic_cloning_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_abortion_test.csv -motion-cross_topic_death_penalty_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_abortion_test.csv -motion-cross_topic_gun_control_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_abortion_test.csv -motion-cross_topic_marijuana_legalization_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_abortion_test.csv -motion-cross_topic_minimum_wage_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_abortion_test.csv -motion-cross_topic_nuclear_energy_dev_abortion_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_abortion_test.csv -motion-cross_topic_school_uniforms_dev_abortion_test ${1:-42}
rm -rd models/*

## cloning
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_cloning_test.csv -motion-cross_topic_abortion_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_cloning_test.csv -motion-cross_topic_death_penalty_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_cloning_test.csv -motion-cross_topic_gun_control_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_cloning_test.csv -motion-cross_topic_marijuana_legalization_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_cloning_test.csv -motion-cross_topic_minimum_wage_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_cloning_test.csv -motion-cross_topic_nuclear_energy_dev_cloning_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_cloning_test.csv -motion-cross_topic_school_uniforms_dev_cloning_test ${1:-42}
rm -rd models/*

## death penalty
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_death_penalty_test.csv -motion-cross_topic_abortion_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_death_penalty_test.csv -motion-cross_topic_cloning_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_death_penalty_test.csv -motion-cross_topic_gun_control_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_death_penalty_test.csv -motion-cross_topic_marijuana_legalization_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_death_penalty_test.csv -motion-cross_topic_minimum_wage_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_death_penalty_test.csv -motion-cross_topic_nuclear_energy_dev_death_penalty_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_death_penalty_test.csv -motion-cross_topic_school_uniforms_dev_death_penalty_test ${1:-42}
rm -rd models/*

## gun control
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_gun_control_test.csv -motion-cross_topic_abortion_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_gun_control_test.csv -motion-cross_topic_cloning_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_gun_control_test.csv -motion-cross_topic_death_penalty_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_gun_control_test.csv -motion-cross_topic_marijuana_legalization_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_gun_control_test.csv -motion-cross_topic_minimum_wage_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_gun_control_test.csv -motion-cross_topic_nuclear_energy_dev_gun_control_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_gun_control_test.csv -motion-cross_topic_school_uniforms_dev_gun_control_test ${1:-42}
rm -rd models/*

## marijuana legalization
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_marijuana_legalization_test.csv -motion-cross_topic_abortion_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_marijuana_legalization_test.csv -motion-cross_topic_cloning_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_marijuana_legalization_test.csv -motion-cross_topic_death_penalty_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_marijuana_legalization_test.csv -motion-cross_topic_gun_control_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_marijuana_legalization_test.csv -motion-cross_topic_minimum_wage_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_marijuana_legalization_test.csv -motion-cross_topic_nuclear_energy_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_marijuana_legalization_test.csv -motion-cross_topic_school_uniforms_dev_marijuana_legalization_test ${1:-42}
rm -rd models/*

## minimum wage
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_minimum_wage_test.csv -motion-cross_topic_abortion_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_minimum_wage_test.csv -motion-cross_topic_cloning_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_minimum_wage_test.csv -motion-cross_topic_death_penalty_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_minimum_wage_test.csv -motion-cross_topic_gun_control_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_minimum_wage_test.csv -motion-cross_topic_marijuana_legalization_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_minimum_wage_test.csv -motion-cross_topic_nuclear_energy_dev_minimum_wage_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_minimum_wage_test.csv -motion-cross_topic_school_uniforms_dev_minimum_wage_test ${1:-42}
rm -rd models/*

## nuclear energy
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_nuclear_energy_test.csv -motion-cross_topic_abortion_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_nuclear_energy_test.csv -motion-cross_topic_cloning_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_nuclear_energy_test.csv -motion-cross_topic_death_penalty_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_nuclear_energy_test.csv -motion-cross_topic_gun_control_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_nuclear_energy_test.csv -motion-cross_topic_marijuana_legalization_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_nuclear_energy_test.csv -motion-cross_topic_minimum_wage_dev_nuclear_energy_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/school_uniforms_dev_nuclear_energy_test.csv -motion-cross_topic_school_uniforms_dev_nuclear_energy_test ${1:-42}
rm -rd models/*

## school uniforms
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/abortion_dev_school_uniforms_test.csv -motion-cross_topic_abortion_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/cloning_dev_school_uniforms_test.csv -motion-cross_topic_cloning_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/death_penalty_dev_school_uniforms_test.csv -motion-cross_topic_death_penalty_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/gun_control_dev_school_uniforms_test.csv -motion-cross_topic_gun_control_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/marijuana_legalization_dev_school_uniforms_test.csv -motion-cross_topic_marijuana_legalization_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/minimum_wage_dev_school_uniforms_test.csv -motion-cross_topic_minimum_wage_dev_school_uniforms_test ${1:-42}
rm -rd models/*
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_cross_topic/nuclear_energy_dev_school_uniforms_test.csv -motion-cross_topic_nuclear_energy_dev_school_uniforms_test ${1:-42}
rm -rd models/*
