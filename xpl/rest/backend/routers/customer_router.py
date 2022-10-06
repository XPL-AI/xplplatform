from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel

from xpl.backend.services import customer_service

router = APIRouter()


class CustomerRegistrationData(BaseModel):
    email: str
    name: str


@router.post("/customer")
async def register_customer(registration_data: CustomerRegistrationData):
    await customer_service.register_customer()