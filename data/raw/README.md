# Raw Dataset

Place the Abaqus/Explicit simulation and validation dataset here:

```txt
data/raw/machining_data.xlsx
```

The workbook must contain these columns:

```txt
rake_angle
nose_radius
grain_size
cutting_speed
RF1
RF2
temperature
```

Example rows:

```csv
rake_angle,nose_radius,grain_size,cutting_speed,RF1,RF2,temperature
5,0.2,fine,0.5,2.0,-2.5,330
5,0.4,medium,0.5,1.7,-2.2,312
-5,0.6,coarse,0.5,1.5,-2.0,270
```
