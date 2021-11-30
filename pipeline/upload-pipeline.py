#!/bin/python3

from datetime import datetime

import git
import kfp

EXPERIMENT = "BTAP ML"
PIPELINE = "Events"


def short_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.commit.hexsha
    return repo.git.rev_parse(sha, short=8)


COMMIT = short_sha()
TIME = str(datetime.now()).split('.')[0]

#################################
# DON'T EDIT:                 ###
# Create the Experiment       ###
#################################
client = kfp.Client()
client._context_setting['namespace']= 'nrcan-btap'
exp = client.create_experiment(
    name=EXPERIMENT,
    namespace='nrcan-btap'
)

# Create pipeline if not exists
try:
    client.upload_pipeline(
        "pipeline.yaml",
        pipeline_name=f"{EXPERIMENT} {PIPELINE}",
        description="BTAP Surrogate model output"
    )
except:
    # Pipeline already exists
    pass

client.upload_pipeline_version(
    "pipeline.yaml",
    pipeline_version_name=f"{TIME} - {COMMIT}",
    pipeline_name=f"{EXPERIMENT} {PIPELINE}"
)

#############################################
# DON'T EDIT:                             ###
# Run the pipeline                        ###
#############################################

run = client.run_pipeline(
    exp.id,
    EXPERIMENT,
    'pipeline.yaml',
)
