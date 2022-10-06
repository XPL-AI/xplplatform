import uuid
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime

import pandas

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import Conflict

from xpl.data import config

BIG_QUERY_PROJECT_ID = config['big_query_project_id']
BIG_QUERY_DATASETS_LOCATION = config['big_query_datasets_location']


class DataItem(BaseModel):
    row_id: Optional[str]
    data_point_id: str
    data_point_file: str
    timestamp: datetime
    concept_id: str
    instance_id: str
    predictor_type: str
    predictor_id: str
    input_set: str
    log_informativeness: float
    logit_confidence: float
    location: Optional[Dict[str, Optional[float]]]
    text: Optional[str]
    value: Optional[float]
    previous_data_point_id: Optional[str]
    next_data_point_id: Optional[str]
    collected_by_device_fingerprint_id: Optional[str]


ALL_LOCATION_COLUMNS = {'center_x', 'center_y', 'center_t', 'half_width', 'half_height', 'half_period'}


class DataItemRepository:
    _client = None

    def __init__(self, dataset_id):
        self.__dataset_id = dataset_id

    def get_client(self) -> bigquery.Client:
        if DataItemRepository._client is None:
            """initialization over system account"""
            DataItemRepository._client = bigquery.Client()
        return DataItemRepository._client

    def table_exists(self, table_name, dataset_id=None, project_id=None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id

        table_id = f'{project_id}.{dataset_id}.{table_name}'

        client = self.get_client()
        try:
            client.get_table(table_id)
            return True
        except NotFound:
            return False

    def create_dataset_if_not_exist(self, 
                                    dataset_id, 
                                    project_id=None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id

        client: bigquery.Client = self.get_client()
        dataset_id = f'{project_id}.{dataset_id}'
        dataset = bigquery.Dataset(dataset_id)

        dataset.location = BIG_QUERY_DATASETS_LOCATION
        client.create_dataset(dataset=dataset_id, exists_ok=True)

    def create_table(self, 
                     table_name: str, 
                     modality_type: str, 
                     dataset_id: str = None, 
                     project_id: str = None,
                     include_text: bool = False,
                     include_value: bool = False):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id

        client: bigquery.Client = self.get_client()
        self.create_dataset_if_not_exist(dataset_id, project_id)

        location_fields = []

        if modality_type == '1d' or modality_type == 'time_series' or modality_type == 'audio':
            location_fields.append(bigquery.SchemaField("center_x", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_width", field_type="FLOAT64", mode="NULLABLE"))
        elif modality_type == '2d' or modality_type == 'image':
            location_fields.append(bigquery.SchemaField("center_x", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("center_y", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_width", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_height", field_type="FLOAT64", mode="NULLABLE"))
        elif modality_type == '3d' or modality_type == 'video':
            location_fields.append(bigquery.SchemaField("center_x", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("center_y", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("center_t", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_width", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_height", field_type="FLOAT64", mode="NULLABLE"))
            location_fields.append(bigquery.SchemaField("half_period", field_type="FLOAT64", mode="NULLABLE"))

        if include_text:
            location_fields.append(bigquery.SchemaField("text", field_type="STRING", mode="NULLABLE"))

        if include_value:
            location_fields.append(bigquery.SchemaField("value", field_type="FLOAT64", mode="NULLABLE"))

        schema = [bigquery.SchemaField("row_id", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("data_point_id", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("data_point_file", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("timestamp", field_type="DATETIME", mode="REQUIRED"),
                  bigquery.SchemaField("concept_id", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("instance_id", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("predictor_type", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("predictor_id", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("input_set", field_type="STRING", mode="REQUIRED"),
                  bigquery.SchemaField("log_informativeness", "FLOAT64", mode="NULLABLE"),
                  bigquery.SchemaField("logit_confidence", "FLOAT64", mode="NULLABLE"),
                  # bigquery.SchemaField("location", field_type='STRUCT', mode="NULLABLE",
                  #                      fields=location_fields),

                  # this field will not be used until we have resources to reliably fingerprint end-user devices.
                  #
                  # Device's fingerprint is usually some combination of MAC, Screen Resolution.. etc selected for every OS
                  # based on available data on the statistical probability to identify device.
                  # There is no reliable way to define it with 100% accuracy. Device/OS vendors don't want to open door
                  # for personal identification.
                  #
                  # When we can do those fingerprints, we can use this field to see all samples that came
                  # from fingerprinted device. At the same time we cannot guarantee that samples belong
                  # to any particular person - therefore we cannot give out samples to anyone base solely on fingerprints.
                  bigquery.SchemaField("collected_by_device_fingerprint_id", field_type="STRING", mode="NULLABLE"),
                  bigquery.SchemaField("previous_data_point_id", field_type="STRING", mode="NULLABLE"),
                  bigquery.SchemaField("next_data_point_id", field_type="STRING", mode="NULLABLE")
                  ]

        schema += location_fields

        table_id = f'{project_id}.{dataset_id}.{table_name}'
        table = bigquery.Table(table_id,
                               schema=schema)

        try:
            client.create_table(table,
                                exists_ok=False)
        except Conflict:
            raise TableExistsException(f'Table table_id={table_id} already exists.')

    def insert(self,
               data_items: list[DataItem],
               table_name: str,
               dataset_id: str = None,
               project_id: str = None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id

        client = self.get_client()

        table_id = f'{project_id}.{dataset_id}.{table_name}'
        table = client.get_table(table_id)

        table_columns = set(c.name for c in table.schema)

        rows_to_insert = []
        for item in data_items:
            row: dict = item.dict()
            if 'location' in row:
                location = row['location']
                if location is not None:
                    row = {**row, **location}
                    del row['location']
            row['row_id'] = str(uuid.uuid4())

            for key in row.keys():
                if key not in table_columns:
                    del row[key]

            rows_to_insert.append(row)

        # BigQuery restricts the size of payload to insert at a time.
        # This can be mitigated with building data pipeline.
        # https://stackoverflow.com/questions/42409983/any-restriction-on-number-of-record-insertion-in-bigquery
        #
        # rows_to_insert = [i.dict() for i in data_items]
        number_of_batches = int(len(rows_to_insert) / 1000) + 1

        errors = []
        for i in range(number_of_batches):
            current_batch = rows_to_insert[i * 1000: i * 1000 + 1000]
            if len(current_batch) > 0:
                errors += client.insert_rows(table, rows_to_insert[i * 1000: i * 1000 + 1000])

        if errors:
            raise Exception(errors)

    def insert_from_data_frame(self,
                               data_frame: pandas.DataFrame,
                               table_name: str,
                               dataset_id: str = None,
                               project_id: str = None):
        """
        Inserts a DataFrame of DataItems to a DataItem-like table in BigQuery.
        DataFrame should exactly follow the schema defined in BigQuery w/o any transformation. This method is a shortcut for bulk inserts,
        when DataFrame structure is already close to the schema.
        @param data_frame: pandas.DataFrame shaped and labeled exactly like schema of DataItem in BigQuery
        @param table_name: a name of the table to insert to.
        @param dataset_id: a BiqQuery dataset to which table belongs.
        @param project_id: a BiqQuery project to which dataset belongs
        """
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id
        client = self.get_client()

        table_id = f'{project_id}.{dataset_id}.{table_name}'
        table = client.get_table(table_id)

        errors = client.insert_rows_from_dataframe(table, data_frame)
        error_count = 0
        for e in errors:
            error_count += len(e)

        if error_count > 0:
            raise Exception(errors)

    def select(self, 
               table_name: str, 
               dataset_id: str = None, 
               project_id: str = None,
               concept_id: str = None,
               predictor_type: str = None,
               predictor_id: str = None,
               input_set: str = None,
               row_ids: List[str] = None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id
        query_job: bigquery.job.QueryJob = self.__select(table_name, dataset_id, project_id,
                                                         concept_id=concept_id,
                                                         predictor_type=predictor_type,
                                                         predictor_id=predictor_id,
                                                         input_set=input_set,
                                                         row_ids=row_ids)

        data_items: list[DataItem] = []
        for row in query_job:
            data_item = DataItem(row_id=row['row_id'],
                                 data_point_id=row['data_point_id'],
                                 data_point_file=row['data_point_file'],
                                 timestamp=row['timestamp'],
                                 concept_id=row['concept_id'],
                                 instance_id=row['instance_id'],
                                 predictor_type=row['predictor_type'],
                                 predictor_id=row['predictor_id'],
                                 input_set=row['input_set'],
                                 log_informativeness=row['log_informativeness'],
                                 logit_confidence=row['logit_confidence'],
                                 location={},
                                 previous_data_point_id=row['previous_data_point_id'],
                                 next_data_point_id=row['next_data_point_id'],
                                 collected_by_device_fingerprint_id=row['collected_by_device_fingerprint_id'])

            location_columns = ALL_LOCATION_COLUMNS.intersection(row.keys())
            for lc in location_columns:
                data_item.location[lc] = row[lc]

            if 'text' in row.keys():
                data_item.text = row['text']

            if 'value' in row.keys():
                data_item.value = row['value']

            data_items.append(data_item)

        return data_items

    def select_as_data_frame(self, table_name: str,
                             dataset_id: str = None,
                             project_id: str = None,
                             concept_id: str = None,
                             predictor_type: str = None,
                             predictor_id: str = None,
                             input_set: str = None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id

        query_job: bigquery.job.QueryJob = self.__select(table_name, dataset_id, project_id,
                                                         concept_id=concept_id,
                                                         predictor_type=predictor_type,
                                                         predictor_id=predictor_id,
                                                         input_set=input_set)

        df: pandas.DataFrame = query_job.to_dataframe()

        return df

    def __select(self, 
                 table_name: str, 
                 dataset_id: str, 
                 project_id: str,
                 concept_id: str = None,
                 predictor_type: str = None,
                 predictor_id: str = None,
                 input_set: str = None,
                 row_ids: List[str] = None) -> bigquery.job.QueryJob:
        client: bigquery.Client = self.get_client()
        table_id = f'{project_id}.{dataset_id}.{table_name}'

        condition = ''
        if row_ids and len(row_ids) > 0:
            condition += f'AND row_id in '
            condition += "('" + "', '".join(row_ids) + "') "
        if concept_id is not None:
            condition += f"AND concept_id='{concept_id}' "
        if predictor_type is not None:
            condition += f"AND predictor_type='{predictor_type}' "
        if predictor_id is not None:
            condition += f"AND predictor_id='{predictor_id}' "
            if input_set is not None:
                condition += f"AND input_set='{input_set}' "

        query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE 1=1
                {condition}
            ORDER BY timestamp
            --LIMIT 20
        """
        query_job: bigquery.job.QueryJob = client.query(query)

        if query_job.error_result is not None:
            raise Exception(query_job.error_result)

        return query_job

    def delete_table(self, 
                     table_name: str, 
                     dataset_id: str = None, 
                     project_id: str = None):
        project_id = BIG_QUERY_PROJECT_ID if project_id is None else project_id
        dataset_id = self.__dataset_id if dataset_id is None else dataset_id

        client: bigquery.Client = self.get_client()
        project_id = project_id if project_id is not None else BIG_QUERY_PROJECT_ID

        table_id = f'{project_id}.{dataset_id}.{table_name}'

        client.delete_table(table_id, not_found_ok=True)


class RepositoryException(Exception):
    """Basic repository exception"""
    pass


class TableExistsException(Exception):
    pass
