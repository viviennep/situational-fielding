import numpy as np, polars as pl, pyarrow as pa
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud import bigquery
cl = pl.col

project = 'your-project-name-here'

bq_client  = bigquery.Client(project=project)

query = f"""
select
 *
from `{project}.baseball_public_dataset.retrosheet_events`
where season_id between 2015 and 2025
"""

job = bq_client.query(query)
arrow_tbl  = job.to_arrow(bqstorage_client=BigQueryReadClient())
df = pl.from_arrow(arrow_tbl).sort('GAME_DATE','EVENT_ID')
df.write_parquet('2015-2024-full-retrosheet.parquet')
df.filter(cl('SEASON_ID').is_in([2021,2022,2023,2024])).write_parquet('2021-2024-full-retrosheet.parquet')



