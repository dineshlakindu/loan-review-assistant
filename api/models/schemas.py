
from __future__ import annotations
from typing import Optional, List, Literal
from decimal import Decimal
from enum import Enum
import re

from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator, constr, conint, confloat

# Sri Lanka NIC: old=9 digits + V/X, new=12 digits. Allow passports (alphanumeric 6–12).
NIC_REGEX_OLD = re.compile(r'^\d{9}[VvXx]$')
NIC_REGEX_NEW = re.compile(r'^\d{12}$')

class EmploymentType(str, Enum):
    SALARIED = 'salaried'
    SELF_EMPLOYED = 'self_employed'
    UNEMPLOYED = 'unemployed'
    STUDENT = 'student'
    RETIRED = 'retired'

class LoanType(str, Enum):
    PERSONAL = 'personal'
    HOME = 'home'
    AUTO = 'auto'
    EDUCATION = 'education'
    BUSINESS = 'business'

class DecisionLabel(str, Enum):
    APPROVE = 'approve'
    REJECT = 'reject'
    FLAG = 'flag'

class LoanApplication(BaseModel):
    """Incoming application payload (for future full model)."""
    model_config = ConfigDict(extra='forbid')

    application_id: constr(strip_whitespace=True, min_length=6, max_length=40)
    customer_id: constr(strip_whitespace=True, min_length=1, max_length=40)
    full_name: constr(strip_whitespace=True, min_length=3, max_length=80)
    nic_or_passport: constr(strip_whitespace=True, min_length=9, max_length=12) = Field(..., description="Sri Lanka NIC (old/new) or passport")
    dob: constr(pattern=r'^\d{4}-\d{2}-\d{2}$') = Field(..., description="YYYY-MM-DD")

    email: Optional[EmailStr] = None
    phone: Optional[constr(strip_whitespace=True, min_length=7, max_length=20)] = None
    country_code: Optional[constr(to_lower=True, min_length=2, max_length=2)] = 'lk'
    address: Optional[constr(strip_whitespace=True, min_length=5, max_length=200)] = None

    loan_type: LoanType
    purpose: constr(strip_whitespace=True, min_length=3, max_length=120)
    loan_amount: Decimal = Field(..., gt=0, description="Requested principal in LKR")
    term_months: conint(ge=6, le=420)
    annual_rate_pct: confloat(ge=0.0, le=60.0) = Field(..., description="Nominal annual interest rate (%)")

    monthly_income: Decimal = Field(..., ge=0)
    monthly_expenses: Decimal = Field(..., ge=0)
    existing_loans_total: Decimal = Field(0, ge=0)
    employment_type: EmploymentType
    employer_name: Optional[constr(strip_whitespace=True, min_length=2, max_length=120)] = None
    credit_history_length_months: conint(ge=0, le=600) = 0
    collateral_value: Optional[Decimal] = Field(default=None, ge=0)

    submitted_at: Optional[constr(pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')] = None

    @field_validator('nic_or_passport')
    @classmethod
    def validate_nic_or_passport(cls, v: str):
        if NIC_REGEX_OLD.match(v) or NIC_REGEX_NEW.match(v):
            return v
        # allow passports (alphanumeric 6–12)
        if re.fullmatch(r'^[A-Za-z0-9]{6,12}$', v):
            return v
        raise ValueError('NIC (old/new) or passport format invalid')

    @property
    def dti_ratio(self) -> float:
        income = float(self.monthly_income)
        obligations = float(self.monthly_expenses + self.existing_loans_total)
        return obligations / income if income > 0 else 1.0

class KYCRecord(BaseModel):
    """What your mock KYC API will return."""
    model_config = ConfigDict(extra='forbid')
    customer_id: str
    watchlist_hit: bool = False
    pep_flag: bool = False
    id_document_valid: bool = True
    address_match_score: confloat(ge=0.0, le=1.0) = 1.0
    aml_risk_score: conint(ge=0, le=100) = 10
    sanctions_sources: List[str] = []

class CreditReport(BaseModel):
    """What your mock credit API will return."""
    model_config = ConfigDict(extra='forbid')
    customer_id: str
    credit_score: conint(ge=300, le=900) = 650
    delinquencies_12m: conint(ge=0, le=50) = 0
    utilization_ratio: confloat(ge=0.0, le=1.0) = 0.2
    inquiries_6m: conint(ge=0, le=30) = 0
    credit_limit_total: Decimal = Field(0, ge=0)
    on_time_payment_rate: confloat(ge=0.0, le=1.0) = 0.95

class RuleHit(BaseModel):
    code: str
    message: str
    severity: Literal['info', 'warn', 'reject', 'flag'] = 'info'

class Decision(BaseModel):
    """Final decision returned by your system."""
    model_config = ConfigDict(extra='forbid')
    label: Literal['approve','reject','flag']
    reasons: List[RuleHit] = []
    confidence: confloat(ge=0.0, le=1.0) = 0.5
    reviewer: Literal['rules', 'llm', 'hybrid'] = 'hybrid'
    trace_id: Optional[str] = None
    application_id: Optional[str] = None

class ReviewBundle(BaseModel):
    """What you’ll feed into the LLM (Step 4)."""
    model_config = ConfigDict(extra='forbid')
    application: LoanApplication
    kyc: KYCRecord
    credit: CreditReport
    preliminary_rules: List[RuleHit] = []
