# Task list

* Make all data pipes work with Localizationodel or LocalizationModelMultilateration
* Make LocalizationModel and LocalizationModelMultilateration work directly with database connections (no data pipes)
* Remove all references to moving_objects in names; fixed objects are anchors
* Consistently call is measurement_value_name or measurement_value_field_name
* Fix up data source in smcmodel so it can be used more than once
* Make shoe sensor and tray sensor subclasses (or otherwise derivative of this module)
* Add functionality to migrate data from current CSV files to DatabaseConnection
* Make visualizations handle database connections directly
* Visualization: show date and time together on x axis
* Visualization: Add standard deviation to plots
* Visualization: Make functions more DRY by breaking out common code
* Replace print statements with logging
* Get better control over shallow vs. deep copies of dataframes
* Add ribbon for standard deviation to plotting functions?
* Add docstrings
* Generate documentation
