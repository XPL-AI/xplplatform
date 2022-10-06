#  Copyright (c) 2021 XPL Technologies AB - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary software
#  Written by Ivan Korol, 2021

from pydantic import BaseModel
from typing import Dict, List

COLUMN_OFFSET = 4


def print_table(items: List[Dict[str, str]], columns: List[str]):
    max_column_values = {}
    for col in columns:
        max_column_values[col] = len(col) + COLUMN_OFFSET

    items_dict = []
    for item in items:
        if isinstance(item, BaseModel):
            item = item.dict()

        items_dict.append(item)
        for col in columns:
            if item[col] is None:
                item[col] = '[Empty]'
            if col in max_column_values:
                if max_column_values[col] < len(item[col]):
                    max_column_values[col] = len(item[col])
            else:
                max_column_values[col] = len(item[col])

    table_str = ''
    header = "    ".ljust(COLUMN_OFFSET + COLUMN_OFFSET)
    for col in columns:
        header += col.ljust(max_column_values[col] + COLUMN_OFFSET)
    table_str += header + '\n'
    for idx, item in enumerate(items_dict):
        row = f'  [{idx}]'.ljust(COLUMN_OFFSET + COLUMN_OFFSET)

        for col in columns:
            row += item[col].ljust(max_column_values[col] + COLUMN_OFFSET)
        row += '\n'
        table_str += row
    print(table_str)
