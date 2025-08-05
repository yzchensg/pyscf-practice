## For water molecules:
### CNN: 
Settings: 1000 epochs, 
- MAE: 1.777497e-01 
- R²: 0.9927

### GNN

- MAE: 2.040656e-01
- R²:  0.9920

## CO2 molecule
### CNN
- MAE: 9.744267e-02 
- R²: 0.9958

### GNN
- MAE: 1.784727e+00
- R²:  -0.3615

## Methanol molecule

### CNN
- MAE: 7.246092e-01
- R²:  0.9271
### GNN
- MAE: 7.790530e-01 
- R²: 0.9218

## Test on invariance (H2O only):

### Rotational Invariance
Only GNN works.
- Test: Rotate around y-axis by 180 degrees
- CNN: 
	- MAE: 1.380992e+02 
	- R²: -2840.8413
	- Trend: shifted upwards; variant, but lost the slope as well;
- GNN: 
	- MAE: 1.923274e-01 
	- R²: 0.9932 
	- Trend: good fit
### Permutational invariance
Only GNN works.
- Test: Random permutation
- CNN: 
	- MAE: 4.650887e+01
	- R²: -322.6473
	- Trend:
- GNN: 
	- MAE: 2.040656e-01
	- R²: 0.9920
	- Trend: perfect fit

### Translational invariance
Both fail.
- Test: move by 10 in x,y,z directions
- CNN: 
	- MAE: 7.468492e+02 
	- R²: -83079.5849; 
	- Trend: line shifted
- GNN: 
	- MAE: 2.614214e+00 
	- R²: -0.7943 
	- Trend: predicting exactly the same value for all sample
