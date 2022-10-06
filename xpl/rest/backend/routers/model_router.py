import json
from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel

from xpl.backend.services import model_service

router = APIRouter()


class ModelConfig(BaseModel):
    customerId: str
    name: str
    modality: str
    tasks: List[str]
    device: str


@router.post("/model")
async def setup(model_config: ModelConfig):
    """not used now. created when project is created"""
    await model_service.setup_new_model(model_config.customerId, model_config.name,
                                        model_config.modality, model_config.output,
                                        model_config.device)

    return model_config


@router.put("/model/{model_id}")
async def update_model(model_setup):
    """"
    This method should be broken down to individual parameters updates since
    every parameter update has different implications
    Some properties will not be possible to update
    """
    return 'Updates model parameters new model'


@router.get("/model/{model_id}")
async def get_model(model_id):
    """Gets the model settings with url to the model"""
    model = await model_service.get_model(model_id)
    del model['modelFileUri']
    model['modelId'] = model_id

    return json.dumps(model)


async def list_models(customer_id):
    """not used now"""
    return ['model_data_1', 'model_data_2']


async def list_models_by_id(model_id):
    """not used now"""
    # 1. fetch model data
    # 2. Verify that model belongs to the Customer. Check their API key for that.
    # 3. Return the URL

    return f'Gets the URI to the {model_id} model. Or fetch the model and return it.'
