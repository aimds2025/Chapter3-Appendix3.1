"""Shared exception hierarchy for the pipeline."""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class AuthorizationError(PipelineError):
    """Raised by Layer 2 / Layer 6 when identity or policy checks fail."""


class SchemaError(PipelineError):
    """Raised by Layer 3 when a record fails schema-registry validation."""


class PoisoningRejected(PipelineError):
    """Raised by Layer 4 when a coreset candidate fails the poisoning guard."""


class PrivacyBudgetExhausted(PipelineError):
    """Raised by the DP budget ledger when a query would exceed epsilon."""


class ComplianceViolation(PipelineError):
    """Raised by Layer 8 when a mandatory control fails."""
