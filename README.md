# CHAMPSTAMP: [CHAM]PSS PulsarS Timing And Monitoring Pipeline

"CHAMPSTAMP" is a pulsar timing and monitoring pipeline developed and used by the [CHIME All-sky Multiday Pulsar Stacking Search (CHAMPSS) Survey](https://github.com/chime-sps). The pipeline is designed for transit radio telescopes (e.g., [CHIME](https://chime-experiment.ca/en)) to process daily pulsar data and to alert for transient events (e.g., sudden brightening events and pulsar glitches).

## Basic Setup and Customization

There is no standardized way to install the pipeline, as it is specifically designed to meet the needs and scientific interests of the CHAMPSS Team. Thus, we strongly recommend you to fork the repository and just modify the source code as needed.

***FOR CHIME COLLABORATION MEMBERS:** This pipeline has been deployed on the Narval Cluster. For more details, please refer to the CHIME Wiki.*

### Requirements

* [PSRCHIVE](https://psrchive.sourceforge.net/)
* [DSPSR](https://dspsr.sourceforge.net/)
* [LaTex](https://www.latex-project.org/get/) \ [TinyTex](https://github.com/rstudio/tinytex) (for generating monitoring reports)
* Other Python packages listed in `requirements.txt` (install using `pip install -r requirements.txt`)

### Workspace Setup

1. Clone the pipeline repository:

   ```bash
   git clone git@github.com:chime-sps/champss_timing.git
   ```

   *If you intend to make significant customizations, consider forking the repository.*

2. Create the `timing_sources` directory in the same location as the pipeline clone:

   ```bash
   mkdir timing_sources
   ```

3. Test the pipeline:

   ```bash
   python champss_timing -h
   ```

### Configuration

1. Initialize the configuration file:

   ```bash
   python champss_timing config -e
   ```

2. Modify the configuration as needed, and test it with:

   ```bash
   python champss_timing config -t
   ```
   
   Slack tokens and JUMP parameters may be left blank and added later. The pulsar name (%PSR%) and glob pattern can be used in data paths.  The following is an example of a configuration:
   
   ```
   {
        "slack_token": {
            "chime": {
                "CHANNEL_ID": "CHANNEL_ID_HERE", 
                "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN_HERE",
                "SLACK_APP_TOKEN": "SLACK_APP_TOKEN_HERE"
            }, 
            "test": {
                "CHANNEL_ID": "CHANNEL_ID_HERE", 
                "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN_HERE",
                "SLACK_APP_TOKEN": "SLACK_APP_TOKEN_HERE"
            }
        }, 
        "backends": {
            "champss": {
                "label": "CHAMPSS Fold Mode", 
                "data_path": "/PATH/TO/CHAMPSS/DATA",
                "jump": [0, 0]
            }, 
            "backend2": {
                "label": "NAME OF THE BACKEND",
                "data_path": "/PATH/TO/BACKEND2/DATA",
                "jump": [<val.>, <unc.>]
            }
        }, 
        "user_defined": {}
    }
   ```

   **NOTE:** Please ensure that each pulsar's data is located in its respective folder (e.g., /path/to/B0525+21; thus, the the data_path will be /path/to/%PSR%/*.ar). Alternatively, modify the code in champss_timing/cli/masterdb.py (search for the line `psr_id = file.split("/")[-2]`) to match your data structure. There is a to-do item fix this restriction, but it is a temporary solution.
   

### Adding Pulsars to the Pipeline

1. Create a subfolder under `timing_sources` for each pulsar:

   ```bash
   mkdir timing_sources/B0525+21
   ```

2. Add the following files to the subfolder:

   * `pulsar.par`: Timing parameters.
   * `paas.std`: Standard template for generating TOAs.
   * `champss_timing.config` (optional): Pulsar-specific configurations (for available configurations, see `champss_timing/backend/pipecore/config.py`).
     Example:
   
     ```
     {
         "ignore_mjds": {
             "earlier_than": 60300
         },
         "settings": {
             "reset_params": false, 
             "use_filters": ["champss"]
         }, 
         "metadata": {
             "tag": "champss"
         }
     }
     ```

### Run the Pipeline

1. Add files to the pipeline master database:

   ```
   python champss_timing masterdb --auto-insert-raw-data
   ```

   The pipeline will read each file in the directories defined in the "data_paths" section of the configuration file. The metadata from each file will be saved to a master database under "timing_sources".

2. Run the main processing pipeline:

   ```
   python champss_timing pipeline
   ```

### Customization

At this point, the pipeline is set up for basic processing (run `python champss_timing -h` to view available functionalities). To fully automate the pipeline for monitoring, consider the following customizations:

* **Custom Scripts:** Add your scripts to the `champss_timing/scripts` directory (e.g., for file transfers). Those scripts can be executed by using `python champss_timing misc <script>`. 
* **Timing Solutions Repository:** Create an empty repository to store timing solutions. Update the configuration file to include the repo and create automation scripts to push changes.
* **Slack Notifications:** Set up a Slack channel for alerts and add the Slack token to the configuration files.
* **Automation:** Schedule the pipeline to run daily using a cronjob or a similar task scheduler:

  ```bash
  crontab -e
  # Add the following line to run the pipeline every day
  0 0 * * * python champss_timing pipeline
  ```

## Final Notes

The installation method above is written in very rough detail, as the project is being developed as it is being used (and has not really been "installed"). Please feel free to reach out if you encounter any problems or need help with the installation. 
