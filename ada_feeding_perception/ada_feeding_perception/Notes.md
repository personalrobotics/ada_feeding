# Notes
## Fork tine positions
Left to Right -->
1. (359, 275)
2. (367, 274)
3. (375, 273)
4. (381, 272)

## Rectangle threshold values
### Trial 1:
* `pt1 = (tine1_x - 50, tine1_y - 75)`
* `pt2 = (tine1_x + 75, tine1_y + 30)`

Note that I don't think we should hardcode the forktine
and rectangle values because if we move the fork, then
it can change. Furthermore, is the sensor specific enough 
where having slighly off positions for the fork tine can cause
issues?

## Depth threshold values
### Trial 1:

## Number of Pixels
### Trial 1:

## Trials
Food on Fork:
```commandline
number:  4774
number:  3997
number:  4847
number:  4091
number:  4069
number:  4091
number:  4125
```
Food not on Fork:
```commandline
number:  4195
number:  956
number:  1102
number:  603
number:  664
number:  4170
number:  4176
number:  4062
```

## Testing to check if initial approach works
1. Home Position
   1. Without Food
   2. With Donut Hole
   3. With leaf
   4. With ___
2. In-Front of face
   1. Without Food
   2. With Donut Hole
   3. With leaf
   4. With ___
3. Over the plate
   1. Without Food
   2. With Donut Hole
   3. With leaf
   4. With ___
4. Edge Case: Straight up
   1. Without Food
   2. With Donut Hole
   3. With leaf
   4. With ___