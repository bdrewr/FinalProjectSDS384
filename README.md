# FinalProjectSDS384

**Goals**

The aim of this project is to perform an exploratory analysis on a dataset consisting of wetted area measurements for various types of packing to develop a general effective area correlation which predicts effective area based on packing parameters and operating conditions. If time allows this will be expanded to generate gas and liquid film mass transfer coefficients correlations.

This is to aid in developing more accurate packing correlations to aid in improving the accuracy and predictive ability of carbon capture process models.

The dataset to be used is ~10000 datapoints collected from bench and pilot-scale absorption column experiments tested on a variety of gas/liquid systems and packing types. Important variables from these runs are solvent concentration, liquid and gas flowrate and temperature, liquid and gas diffusivities, packing surface area, corrugation angles and channel dimensions. The primary sources of data are from the PhD dissertations of Tsai (2010), Wang (2015), and Song (2017). Supplemental sources of data include the papers of Zakeri (2011) and X-Ray tomography data of absorption column experiments.

The dataset will be cleaned and preprocessed for regression analysis via Support Vector Regression, Random Forest Trees, and Artificial Neural Networks to construct correlations for Fractional Wetted Area, Pressure Drop and kL.

These correlations will be tested on test datasets for accuracy in predictions, and against statistical methods such as R-Squared, RMSE, and MAPE.

References:
Abreu, M. et al. CO2 absorber intensification by high liquid flux operation. GHGT-16. 2022.
Hazare, S.R. et al. Correlating Interfacial Area and Volumetric Mass Transfer Coefficient in Bubble Column with the Help of Machine Learning Methods. IeCR. 2022.
Song, D. Effect of Liquid Viscosity on Liquid Film Mass Transfer for Packings. PhD Dissertation. 2017.
Song, D. et al. Mass Transfer Parameters for Packings: Effect of Viscosity. IeCR. 2018.
Tsai, R. Mass Transfer Area of Structured Packing. PhD Dissertation. 2010.
Wang, C. Mass Transfer Coefficients and Effective Area of Packing. PhD Dissertation. 2015.
Zakeri, A. et al. Experimental Investigation of Pressure Drop, Liquid Hold-Up and Mass Transfer Parameters in a 0.5 m Diameter Absorber Column. GHGT-10. 2011.
