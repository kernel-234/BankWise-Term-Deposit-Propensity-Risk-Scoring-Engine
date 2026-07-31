from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    age: int = Field(..., example=35, description="Age of the customer")
    job: str = Field(..., example="technician", description="Type of job")
    marital: str = Field(..., example="single", description="Marital status")
    education: str = Field(..., example="tertiary", description="Education level")
    default: str = Field(..., example="no", description="Has credit in default?")
    balance: float = Field(..., example=1350.0, description="Average yearly balance")
    housing: str = Field(..., example="yes", description="Has housing loan?")
    loan: str = Field(..., example="no", description="Has personal loan?")
    contact: str = Field(..., example="cellular", description="Contact communication type")
    day: int = Field(..., example=15, description="Last contact day of the month")
    month: str = Field(..., example="may", description="Last contact month")
    duration: int = Field(..., example=320, description="Last contact duration in seconds")
    campaign: int = Field(..., example=2, description="Number of contacts during this campaign")
    pdays: int = Field(..., example=-1, description="Days passed after previous campaign")
    previous: int = Field(..., example=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., example="unknown", description="Outcome of previous marketing campaign")

class PredictionResponse(BaseModel):
    conversion_probability: float
    predicted_class: int
    recommendation: str
    optimal_threshold: float