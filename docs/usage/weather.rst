EnergyPlus uses weather data in the EPW file format, as defined in the `Auxiliary
Programs <https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v9.6.0/AuxiliaryPrograms.pdf>`_ documentation. In
order to be consumed by the model, the data needs to undergo several steps before it can be used.

The weather data is received from a `GitHub repo <https://raw.githubusercontent.com/NREL/openstudio-standards/nrcan/data/weather/>`_.

Weather data can be automatically preprared from the information found in the input building .xlsx files. The
``prepare_weather.py`` script can be used to process and save weather data.

Although ``prepare_weather.py`` can be used to process the weather files directly, within the training and
running pipelines, the weather is processed as part of the data preprocessing step by loading all weather
files attached to buildings in the input file(s). The contents of this subsection outline how the weather data
is processed.

.. note::

    The weather data is only loaded for energy preprocessing and if there are connection issues, the code may
    return an error when retrieving the weather from the repo. You can try running the program again if this
    arises.

.. code::

    python prepare_weather.py input_config.yml

.. note::

   The input to ``prepare_weather.py`` is the same configuration file that is given to ``train_model_pipeline.py``
   and to ``run_model.py``. The CLI can also be used to invoke the process without a complete configuration file.
   The weather files to process must now be passed through the CLI since it is no longer an input in the
   configuration file.

Outputs are placed into a ``/weather`` folder (by default) in storage as parquet files. The name of the weather files
will match the input name found in the YAML file, but with a ``.parquet`` file extension.

.. note::

    When called within the data preprocessing step, no ``.parquet`` file is output. The output is directly used without
    needing to be loaded again.

What this does
^^^^^^^^^^^^^^

EPW files contain several lines of data that is irrelevant to the model. This data is all in the first several lines
of the file. For example, everything before the line starting with ``1966`` can be ignored::

    LOCATION,Montreal Int'l,PQ,CAN,WYEC2-B-94792,716270,45.47,-73.75,-5.0,36.0
    DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-23.7,-21.1,-30.5,0.2,-23,-27.9,0.3,-20.6,12.9,-5.3,11.5,-7.9,3.9,260,Cooling,7,9.3,30,22.1,28.5,21.1,27.1,20.2,23.2,28.1,22.2,26.6,21.4,25.6,4.9,220,21.6,16.3,26,20.7,15.5,25.2,19.8,14.5,24.2,69.3,28.1,65.5,26.7,62.3,25.6,703,Extremes,11.1,9.7,8.6,27.4,-26.5,32.3,2.9,1.5,-28.6,33.4,-30.4,34.3,-32,35.2,-34.2,36.3
    TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,7/13,7/19,Summer - Week Nearest Average Temperature For Period,Typical,6/ 8,6/14,Winter - Week Nearest Min Temperature For Period,Extreme,1/ 6,1/12,Winter - Week Nearest Average Temperature For Period,Typical,2/17,2/23,Autumn - Week Nearest Average Temperature For Period,Typical,10/13,10/19,Spring - Week Nearest Average Temperature For Period,Typical,4/12,4/18
    GROUND TEMPERATURES,3,.5,,,,-1.50,-6.19,-7.46,-6.35,-0.03,7.05,13.71,18.53,19.94,17.67,12.21,5.33,2,,,,2.71,-1.68,-3.77,-3.85,-0.51,4.33,9.54,14.01,16.32,15.89,12.81,8.08,4,,,,5.45,2.05,-0.04,-0.69,0.54,3.36,6.87,10.31,12.62,13.17,11.85,9.08
    HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0
    COMMENTS 1,WYEC2-Canadian Weather year for Energy Calculations (CWEC) -- WMO#716270
    COMMENTS 2, -- Ground temps produced with a standard soil diffusivity of 2.3225760E-03 {m**2/day}
    DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
    1966,1,1,1,60,A_A__*A_9*M_Q_M_Q_Q_Q_9_____________*_*___*,6.8,4.9,88,100550,0,9999,322,0,0,0,0,0,0,999900,225,7.2,10,10,16.1,3600,0,999999999,0,0.0000,0,88,0.000,0.0,0.0
    1966,1,1,2,60,A_A__*A_9*M_Q_M_Q_Q_Q_9_____________*_*___*,8.3,6.2,87,100310,0,9999,330,0,0,0,0,0,0,999900,248,6.7,10,10,16.1,3600,0,999999999,0,0.0000,0,88,0.000,0.0,0.0
    1966,1,1,3,60,A_A__*A_9*M_Q_M_Q_Q_Q_9_____________*_*___*,9.2,7.1,87,100170,0,9999,335,0,0,0,0,0,0,999900,248,8.1,10,10,16.1,3600,0,999999999,0,0.0000,0,88,0.000,0.0,0.0

Column names for the tabular data can be found in the Auxiliary Programs documentation.

Remove unused columns
"""""""""""""""""""""

In the Auxiliary Programs documentation for the EPW file format it lists which columns in the file are actually used
for predicting building energy use. Columns marked as not used by EnergyPlus are removed from the dataset.

.. note::

   The ``year`` column is marked as not being used by EnergyPlus, but is kept for use in creating a datetime index
   of the data. Only the month and day values are used when merging with building and energy data.

Summarize by day
""""""""""""""""

Weather data in the file is provided for every hour for all 365 days of the year. Due to the way the data is combined
in later steps, this becomes millions of records that need to be processed despite the fact that there is no intent to
predict energy use to the hourly level. To make working with the data more manageable, weather values are summarized to
daily intervals.
