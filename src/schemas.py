from pydantic import BaseModel, Field
from typing import Literal


class CustomerRequestData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen : Literal[0, 1]
    Partner : Literal['Yes', 'No']
    Dependents : Literal['Yes', 'No']
    tenure : int = Field(ge=0, le=72)
    PhoneService : Literal['Yes', 'No']
    MultipleLines : Literal['Yes','No','No phone service']
    InternetService : Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity : Literal['No', 'Yes', 'No internet service']
    OnlineBackup : Literal['Yes', 'No', 'No internet service']
    DeviceProtection : Literal['Yes', 'No', 'No internet service']
    TechSupport : Literal['Yes', 'No', 'No internet service']
    StreamingTV : Literal['Yes', 'No', 'No internet service']
    StreamingMovies : Literal['Yes', 'No', 'No internet service']
    Contract : Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling : Literal['Yes','No']
    PaymentMethod : Literal['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']
    MonthlyCharges : float = Field(ge=0, le=200)
    TotalCharges : float = Field(ge=0, le=10000)

class PredictionResponseModel(BaseModel):
    prediction : Literal["Churn", "Not Churn"]
    probability : float