{
  "family": "lead69_horizon_two_look",
  "source": "docs/lead69_results.md",
  "classification": "unresolved_below_power",
  "league": "nfl",
  "season_start": 2023,
  "season_end": 2025,
  "category": "schedule",
  "classification_evidence": "Three season blocks; no admissible closure ground; calibration worsens. Two declared looks; diagnostic cells are correlated.",
  "notes": "799 starting games; 114 structural SNF/MNF exclusions; 685 scored. Earliest predeadline captured kickoff. Accuracy units are percentage points. Do not pool unlike metrics. Registry pending orchestrator.",
  "cells": [
    {
      "name": "lead69_horizon_loso_accuracy",
      "description": "LEAD-69 horizon: paired OOS accuracy gain versus refitted four-term base; descriptive if slots.",
      "effect": 0.291970802919708,
      "effect_units": "accuracy_points",
      "interval_low": -0.24853801169590642,
      "interval_high": 0.7794377888398026,
      "probability_positive": 0.7962962962962963,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_horizon_loso_log_loss",
      "description": "LEAD-69 horizon: paired OOS log_loss gain versus refitted four-term base; descriptive if slots.",
      "effect": -0.0009797746565871816,
      "effect_units": "log_loss_improvement",
      "interval_low": -0.002284630862748496,
      "interval_high": 0.00011037155808688822,
      "probability_positive": 0.14814814814814814,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_horizon_loso_brier",
      "description": "LEAD-69 horizon: paired OOS brier gain versus refitted four-term base; descriptive if slots.",
      "effect": -0.0004355076410244651,
      "effect_units": "brier_improvement",
      "interval_low": -0.0010165275719631838,
      "interval_high": 5.4996173605733725e-05,
      "probability_positive": 0.14814814814814814,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_horizon_loso_line_move",
      "description": "LEAD-69 horizon: paired OOS line_move gain versus refitted four-term base; descriptive if slots.",
      "effect": 0.008759124087591242,
      "effect_units": "ats_points",
      "interval_low": 0.00533625730994152,
      "interval_high": 0.012165327765541648,
      "probability_positive": 1.0,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_slots_loso_accuracy",
      "description": "LEAD-69 slots: paired OOS accuracy gain versus refitted four-term base; descriptive if slots.",
      "effect": 0.583941605839416,
      "effect_units": "accuracy_points",
      "interval_low": -0.34328979382763475,
      "interval_high": 1.7164489691381721,
      "probability_positive": 0.7222222222222222,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_slots_loso_log_loss",
      "description": "LEAD-69 slots: paired OOS log_loss gain versus refitted four-term base; descriptive if slots.",
      "effect": -0.009418119385659398,
      "effect_units": "log_loss_improvement",
      "interval_low": -0.024736833360525824,
      "interval_high": 0.00453141219531714,
      "probability_positive": 0.14814814814814814,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_slots_loso_brier",
      "description": "LEAD-69 slots: paired OOS brier gain versus refitted four-term base; descriptive if slots.",
      "effect": -0.0021940856523113856,
      "effect_units": "brier_improvement",
      "interval_low": -0.006086763220432313,
      "interval_high": 0.0017366447591596103,
      "probability_positive": 0.14814814814814814,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    },
    {
      "name": "lead69_slots_loso_line_move",
      "description": "LEAD-69 slots: paired OOS line_move gain versus refitted four-term base; descriptive if slots.",
      "effect": -0.04671532846715328,
      "effect_units": "ats_points",
      "interval_low": -0.10685958725349154,
      "interval_high": 0.009576023391812825,
      "probability_positive": 0.14814814814814814,
      "sample_games": 685,
      "sample_blocks": 3,
      "plain_summary": "Giving line movement different weight by time before kickoff changes a few picks; these seasons do not settle whether it helps."
    }
  ]
}
