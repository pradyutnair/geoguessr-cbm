# Concept Analysis (Stage 2 Pipeline)

## Summary
This folder contains the outputs from the Stage 2 concept analysis run that evaluated Stage 1 concept predictions on the GeoGuessr dataset using the local HF cache.

**Run characteristics**
- Dataset source: Local HF cache (non-360 + 360 JPEGs), missing files filtered.
- Split: train
- Max samples requested: 20,000
- Effective evaluated samples: **3,146**
- Countries represented: **36**
- Unique concepts predicted: **761**
- Unique parent concepts predicted: **173**

## Key Metrics
- Average concept confidence: **0.1211**
- Average parent confidence: **0.2933**

**Prediction concentration**
- Top 1 concept: **2.5%** of predictions
- Top 5 concepts: **9.8%**
- Top 10 concepts: **16.3%**
- Top 20 concepts: **23.6%**

## Top Predicted Concepts (Top 20)
1. Bollard — n=78 (2.5%), conf=0.167
2. Roadlines — n=78 (2.5%), conf=0.276
3. Unique Car — n=52 (1.7%), conf=0.181
4. Rifts — n=51 (1.6%), conf=0.145
5. Unique car — n=49 (1.6%), conf=0.197
6. Mountains — n=47 (1.5%), conf=0.116
7. Ten Digit Yellow Pole Codes — n=45 (1.4%), conf=0.238
8. Architecture — n=39 (1.2%), conf=0.114
9. Chevrons — n=37 (1.2%), conf=0.154
10. trident — n=36 (1.1%), conf=0.081
11. Chilo Island — n=33 (1.0%), conf=0.175
12. Bollards — n=29 (0.9%), conf=0.157
13. Baden-Wrttemberg — n=26 (0.8%), conf=0.120
14. Architecture - Black Shingle Roof — n=25 (0.8%), conf=0.098
15. Landscape — n=25 (0.8%), conf=0.128
16. Catalan Language — n=19 (0.6%), conf=0.168
17. Vibes — n=19 (0.6%), conf=0.159
18. NSW guardrail ending — n=19 (0.6%), conf=0.152
19. Posadas — n=18 (0.6%), conf=0.124
20. Red Soil — n=18 (0.6%), conf=0.118

## Least Predicted Concepts (Bottom 20 of those predicted at least once)
1. Yellow lamp — n=1 (0.03%), conf=0.028
2. Peru Copyright 2024 — n=1 (0.03%), conf=0.120
3. Ancash Gen 4 Landscape — n=1 (0.03%), conf=0.062
4. Painted Poles — n=1 (0.03%), conf=0.060
5. SA flat sign posts — n=1 (0.03%), conf=0.041
6. Lake Victoria-SA Border — n=1 (0.03%), conf=0.061
7. License plate — n=1 (0.03%), conf=0.037
8. Infrastructure - Yellow Tuk Tuk / Black — n=1 (0.03%), conf=0.233
9. Las Palmas Bollard — n=1 (0.03%), conf=0.087
10. June/July | No Car (No Antenna) — n=1 (0.03%), conf=0.061
11. Brick houses — n=1 (0.03%), conf=0.098
12. New Forest landscape — n=1 (0.03%), conf=0.175
13. West Mannheim Architecture — n=1 (0.03%), conf=0.085
14. Soybeans — n=1 (0.03%), conf=0.076
15. Road marker — n=1 (0.03%), conf=0.028
16. Fruit orchards — n=1 (0.03%), conf=0.093
17. Cologne Lamppost Markings — n=1 (0.03%), conf=0.224
18. Commie blocks — n=1 (0.03%), conf=0.106
19. Tall hills - Darker soil — n=1 (0.03%), conf=0.032
20. Vic pole top — n=1 (0.03%), conf=0.077

## Parent Concept Distribution (Top 15)
1. building_facade — n=168 (5.3%), children=53
2. bollard_design — n=115 (3.7%), children=38
3. road_sign — n=114 (3.6%), children=56
4. script_latin — n=112 (3.6%), children=59
5. camera_meta — n=104 (3.3%), children=47
6. pole_marker — n=97 (3.1%), children=28
7. road_line — n=89 (2.8%), children=17
8. car_meta — n=88 (2.8%), children=37
9. vegetation_forest_temperate — n=80 (2.5%), children=47
10. landscape_hills_temperate — n=78 (2.5%), children=46
11. chevron — n=76 (2.4%), children=30
12. landscape_hills_tropical — n=71 (2.3%), children=42
13. pole_shape — n=69 (2.2%), children=32
14. car_type — n=66 (2.1%), children=18
15. landscape_mountains_temperate — n=63 (2.0%), children=25

## Predictions by Country (Top 15)
1. AE — n=120, top concept: Roadlines
2. AL — n=120, top concept: Rifts
3. AR — n=120, top concept: San Luis
4. AT — n=120, top concept: Bollard
5. EC — n=120, top concept: Esmeraldas Province
6. AU — n=120, top concept: NSW guardrail ending
7. BD — n=120, top concept: Southern tree stands
8. BE — n=120, top concept: Architecture - Black Shingle Roof
9. BO — n=120, top concept: City Roads, Santa Cruz &
10. BG — n=120, top concept: Wind turbines
11. BT — n=120, top concept: Architecture
12. BR — n=120, top concept: Red Soil
13. CH — n=120, top concept: Low cam
14. CL — n=120, top concept: Chilo Island
15. BW — n=120, top concept: Landscape

## Files in This Folder
- [analysis_report.txt](analysis_report.txt) — full text report (source for the figures above)
- [predictions.csv](predictions.csv) — per-sample predictions (concept + parent + top-5)
- [concept_frequency.csv](concept_frequency.csv) — per-concept counts and confidence stats
- [parent_frequency.csv](parent_frequency.csv) — parent concept distribution
- [country_analysis.csv](country_analysis.csv) — per-country prediction stats
- [_stat.txt](_stat.txt) — report timestamp

## Notes
- The run used cache-only data to avoid HF rate limits; missing image files were filtered prior to inference.
- The plots were regenerated with a report-friendly theme and saved as PNG and PDF in the results directory during the run.