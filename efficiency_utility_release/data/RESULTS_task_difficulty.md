# Task difficulty reference — all 58 benchmark tasks

SR, pass@3, pass^3 pooled across all 24 tier-pool models (72 trials/task: 24 models x 3
trials each). Success is decided by the finalized 3-LLM judge panel (hy3-preview + Seed
2.0 Lite + Grok 4.3): each judge is a 3-pass majority over its base/_e2/_e3 re-evaluations,
and a trajectory passes on a 2-of-3 majority across the three judges — the same panel that
gates every BT/IAA/listwise pair (see RESULTS_ablation.md). Sorted hardest -> easiest by
pooled SR.
`pass@3` = P(at least 1 of 3 trials succeeds, per model, averaged across models).
`pass^3` = P(all 3 trials succeed, per model, averaged across models).

Reproduce: `python3 build_task_difficulty_table.py`.

44 rh44 tasks + 14 custom14 tasks = 58 total, all with full 72-trial coverage.

| Task | Domain | n | SR% | pass@3 | pass^3 |
|---|---|---|---|---|---|
| lc_04_cd_taxi_pref_dismiss | rh44 | 72 | 12.5 | 0.250 | 0.042 |
| custom_028_shanghai_station_sightseeing_tour_reminder | custom14 | 72 | 18.1 | 0.292 | 0.083 |
| custom_029_chongqing_family_tour_plan | custom14 | 72 | 18.1 | 0.292 | 0.083 |
| mt_08_bj_pet_supermarket_weekend | rh44 | 72 | 18.1 | 0.375 | 0.042 |
| rp_03_cd_furniture_moving | rh44 | 72 | 23.6 | 0.417 | 0.125 |
| mt_07_gz_pickup_classmate_dinner | rh44 | 72 | 29.2 | 0.500 | 0.125 |
| rp_02_km_errands_catch_train | rh44 | 72 | 29.2 | 0.500 | 0.125 |
| rp_04_tj_four_stops_errands | rh44 | 72 | 29.2 | 0.583 | 0.083 |
| airport_new_05_sjz_via_friend_pm5 | rh44 | 72 | 34.7 | 0.542 | 0.167 |
| custom_031_guangzhou_tourist_sightseeing_cantonese | custom14 | 72 | 36.1 | 0.500 | 0.250 |
| mt_09_nj_business_three_stop | rh44 | 72 | 36.1 | 0.500 | 0.167 |
| lc_05_hz_elder_hospital_smooth | rh44 | 72 | 38.9 | 0.625 | 0.208 |
| rp_01_sh_kid_clinic_class | rh44 | 72 | 38.9 | 0.583 | 0.167 |
| rp_05_hz_elder_checkup_nohist | rh44 | 72 | 38.9 | 0.625 | 0.208 |
| lc_02_bj_kid_hospital_scheduled | rh44 | 72 | 43.1 | 0.625 | 0.208 |
| custom_018_guangzhou_student_shangjiu_preference | custom14 | 72 | 45.8 | 0.583 | 0.292 |
| lc_03_sh_business_flight_luggage | rh44 | 72 | 47.2 | 0.625 | 0.292 |
| mt_07b_gz_pickup_classmate_dinner_scheduled | rh44 | 72 | 47.2 | 0.708 | 0.250 |
| lc_01_sz_dismiss_home_no_ev | rh44 | 72 | 50.0 | 0.625 | 0.417 |
| lc_09_nj_business_hotel_client | rh44 | 72 | 50.0 | 0.625 | 0.375 |
| lc_11_hz_after_work_via_supermarket | rh44 | 72 | 51.4 | 0.667 | 0.375 |
| ft_014_counterintuitive_ambiguous_time | rh44 | 72 | 52.8 | 0.792 | 0.250 |
| airport_new_04_sz_cheapest_company | rh44 | 72 | 54.2 | 0.750 | 0.208 |
| lc_10_cq_pickup_luxury | rh44 | 72 | 54.2 | 0.625 | 0.458 |
| rp_05_hz_elder_checkup | rh44 | 72 | 54.2 | 0.750 | 0.292 |
| rp_09_cq_pickup_parents_hotpot | rh44 | 72 | 54.2 | 0.750 | 0.375 |
| rp_08_nj_pickup_classmates_dinner | rh44 | 72 | 55.6 | 0.792 | 0.375 |
| rp_10_xa_pickup_friends_dropoff | rh44 | 72 | 55.6 | 0.667 | 0.417 |
| lc_08_wh_late_night_safety | rh44 | 72 | 56.9 | 0.708 | 0.417 |
| rp_06_sz_bank_loop_quiet | rh44 | 72 | 59.7 | 0.750 | 0.417 |
| rp_11_wh_pickup_parents_friend_dropoff | rh44 | 72 | 59.7 | 0.875 | 0.333 |
| rp_07_su_cake_loop | rh44 | 72 | 62.5 | 0.708 | 0.542 |
| custom_024_shenzhen_home_pizzahut_complaint_chitchat | custom14 | 72 | 63.9 | 0.750 | 0.500 |
| custom_016_beijing_tourist_shisanling_preference | custom14 | 72 | 65.3 | 0.875 | 0.417 |
| custom_023_shanghai_tangchenyipin_disney_long_chitchat | custom14 | 72 | 65.3 | 0.792 | 0.500 |
| custom_026_guangzhou_huangpu_home_south_station_family_trip | custom14 | 72 | 65.3 | 0.792 | 0.500 |
| mt_18_bj_short_stop_waiting | rh44 | 72 | 65.3 | 0.792 | 0.542 |
| airport_new_01_bj_home_flight | rh44 | 72 | 69.4 | 0.792 | 0.542 |
| exp_090_extreme_temporal_backward | rh44 | 72 | 69.4 | 0.792 | 0.583 |
| airport_new_03_gz_quick_t2_pm3 | rh44 | 72 | 70.8 | 0.833 | 0.625 |
| ft_021_time_contradict_current | rh44 | 72 | 72.2 | 0.833 | 0.542 |
| mt_01_sh_kid_school_commute | rh44 | 72 | 72.2 | 0.833 | 0.583 |
| custom_025_shenzhen_huaqiangbei_north_station_biztrip_chitchat | custom14 | 72 | 73.6 | 0.833 | 0.583 |
| custom_017_beijing_tourist_yiheyuan_female_driver | custom14 | 72 | 75.0 | 0.833 | 0.667 |
| lc_06_gz_morning_rush_immediate | rh44 | 72 | 75.0 | 0.917 | 0.583 |
| mt_04_hz_flower_after_work | rh44 | 72 | 75.0 | 0.833 | 0.625 |
| mt_06_cd_elder_pharmacy_hospital | rh44 | 72 | 75.0 | 0.875 | 0.542 |
| custom_022_guangzhou_home_baitiange_long_chitchat | custom14 | 72 | 76.4 | 0.833 | 0.667 |
| mt_05_sz_morning_coffee_client | rh44 | 72 | 76.4 | 0.875 | 0.625 |
| exp_056_combo_compound_temporal | rh44 | 72 | 77.8 | 0.833 | 0.750 |
| lc_07_sz_weekend_gym_smooth | rh44 | 72 | 77.8 | 0.917 | 0.625 |
| supp_c1_002_8acf69 | rh44 | 72 | 79.2 | 0.833 | 0.708 |
| custom_027_beijing_shuangqing_military_museum_chitchat | custom14 | 72 | 80.6 | 0.875 | 0.708 |
| mt_03_sh_pickup_two_friends | rh44 | 72 | 80.6 | 0.875 | 0.708 |
| mt_02_bj_hotel_pickup_friend | rh44 | 72 | 83.3 | 0.917 | 0.792 |
| custom_015_shanghai_mcdonalds_rain_nosmoking | custom14 | 72 | 84.7 | 0.875 | 0.792 |
| custom_030_beijing_programmer_recurring_wednesday_ride | custom14 | 72 | 84.7 | 0.875 | 0.792 |
| airport_new_02_xm_current_immediate | rh44 | 72 | 87.5 | 0.917 | 0.833 |