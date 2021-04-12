#!/usr/bin/env python3

""" Constants & Experiments. """

EXPERIMENTS = {
    'All': {
        'Simple': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop_wBGT': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_wBGTraffic_deep100_1000_128',
        # 'RTT_ext': 'ppo_1000ag_5m_wParetoDistr_30_2_30_2_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext',
        'RTT_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTTCoop_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
    },
    '30_2': {
        'Simple': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTTCoop_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
    },
    '30_3': {
        'Simple': 'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT': 'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop': 'ppo_1000ag_5m_wParetoDistr_30_3_GlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTTCoop_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
    },
    '45_4': {
        'Simple': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT_noPTW': 'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTTCoop_noPTW': 'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
    },
    '60_5': {
        'Simple': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        # 'RTT_ext': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext',
        'RTT_noPTW': 'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTTCoop_noPTW': 'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        'RTT_carOnly': 'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly',
        'RTTCoop_carOnly': 'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly',
        'RTT_metroOnly': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly',
        'RTTCoop_metroOnly': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly',
        'RTT_wParking': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking',
        'RTTCoop_wParking': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking',
        # 'RTT_wOwnership': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        # 'RTTCoop_wOwnership': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT_wOwn_noMask': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noMask',
        'RTTCoop_wOwn_noMask': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noMask',
        'RTT_wPreferences': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTTCoop_wPreferences': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'RTT_wPref_byChoice': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_byChoice',
        'RTTCoop_wPref_byChoice': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_byChoice',
    },
}

WAITING_M = 402.17 / 60.0
LATENESS_M = 761.81 / 60.0

NUM_LATE = 595.43

ARRIVAL_H = 32715.66 / 3600.0
DEPARTURE_H = 31613.13 / 3600.0

TRAVEL_TIME_M = 1102.53 / 60.0

EXPERIMENTS_BY_REWARD = {
    'wSimplifiedReward_noBGTraffic': {
        'complete': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_2': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_3': 'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '45_4': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    },
    'wSimpleTTReward_noBGTraffic': {
        'complete': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'complete_ext': 'ppo_1000ag_5m_wParetoDistr_30_2_30_2_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext',
        'complete_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '30_2': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_2_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '30_3': 'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_3_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '45_4': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '45_4_noPTW': 'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '60_5': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_ext': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext',
        '60_5_noPTW': 'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '60_5_carOnly': 'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly',
        '60_5_metroOnly': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly',
        '60_5_wParking': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking',
        '60_5_wOwnership': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_wOwn_noMask': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noMask',
        '60_5_wPreferences': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_wPref_byChoice': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_byChoice',
    },
    'wSimpleTTCoopReward_noBGTraffic': {
        'complete': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'complete_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '30_2': 'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_2_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '30_3': 'ppo_1000ag_5m_wParetoDistr_30_3_GlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '30_3_noPTW': 'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '45_4': 'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '45_4_noPTW': 'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '60_5': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_noPTW': 'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW',
        '60_5_carOnly': 'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly',
        '60_5_metroOnly': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly',
        '60_5_wParking': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking',
        '60_5_wOwnership': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_wOwn_noMask': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noMask',
        '60_5_wPreferences': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        '60_5_wPref_byChoice': 'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_byChoice',
    },
    'wSimpleTTCoopReward_wBGTraffic': {
        'complete': 'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_wBGTraffic_deep100_1000_128',
    }
}
