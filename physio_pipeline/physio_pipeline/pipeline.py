"""
physio_pipeline.pipeline
========================

The orchestrator. Wires the eight layers together with the shared cross-cutting
services and runs a batch end-to-end. Reading `run_batch()` shows the exact
layer interactions and which contract crosses each boundary.

    L1 EdgeDevice.emit()      -> EdgePacket
    L2 Perimeter.authorize()  -> EdgePacket(authorized)
    L3 Ingestion.produce()    -> StreamRecord         (then consume())
    L4 Processor.process()     -> ProcessedBatch
    L5 Storage.persist()      -> StorageReceipt
    L6 MCPGateway.call_tool()  -> gated agent response  (reads L5)
    L7 Training.train()        -> ModelArtifact         (reads L5 coreset)
    L8 Governance.attest()     -> ComplianceReport      (reads audit + ledger)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .core.contracts import ComplianceReport, ModelArtifact, ProcessedBatch, StorageReceipt
from .crosscutting.audit import AuditLog
from .crosscutting.iam import IAM
from .crosscutting.privacy_budget import PrivacyBudgetLedger
from .crosscutting.tracing import Tracer
from .layer1_edge import EdgeDevice
from .layer2_perimeter import ZeroTrustPerimeter
from .layer3_ingestion import StreamIngestion
from .layer4_processing import StreamProcessor
from .layer5_storage import StorageLayer
from .layer6_mcp_gateway import MCPGateway
from .layer7_training import TrainingPipeline
from .layer8_governance import GovernanceLayer


@dataclass
class BatchResult:
    receipt: StorageReceipt
    batch: ProcessedBatch
    agent_response: dict
    model: ModelArtifact
    compliance: ComplianceReport
    trace: list[tuple[str, float]]


class PhysioPipeline:
    def __init__(self, total_epsilon: float = 5.0):
        # --- cross-cutting services shared by all layers ---
        self.audit = AuditLog()
        self.iam = IAM()
        self.ledger = PrivacyBudgetLedger(total_epsilon=total_epsilon)
        self.tracer = Tracer()

        # --- layers ---
        self.perimeter = ZeroTrustPerimeter(self.iam, self.audit)
        self.ingestion = StreamIngestion(self.audit)
        self.processor = StreamProcessor(self.audit)
        self.storage = StorageLayer(self.ledger)
        self.gateway = MCPGateway(self.iam, self.audit)
        self.training = TrainingPipeline(self.storage, self.ledger, self.audit)
        self.governance = GovernanceLayer(self.audit, self.ledger)

        # register the AI agent identity used at Layer 6
        self.iam.register("spiffe://agents/clinical-copilot", {"clinician_ro"})
        self.iam.register("spiffe://ml/trainer", {"ml_trainer"})

    def register_device(self, device: EdgeDevice) -> None:
        self.iam.register(f"spiffe://edge/{device.device_id}", {"device"})

    def run_batch(self, devices: list[EdgeDevice], windows_per_device: int = 4,
                  run_id: str | None = None) -> BatchResult:
        run_id = run_id or uuid.uuid4().hex[:8]

        # ---------- L1 + L2 + L3: acquire, authorize, ingest ----------
        with self.tracer.span("L1-L3.acquire_authorize_ingest"):
            for dev in devices:
                self.register_device(dev)
                for _ in range(windows_per_device):
                    frame = dev.acquire()               # L1 acquire
                    packet = dev.emit(frame)            # L1 on-device DSP
                    packet = self.perimeter.authorize(packet)  # L2
                    self.ingestion.produce(packet)     # L3

        # ---------- L4: stream processing ----------
        with self.tracer.span("L4.process"):
            records = self.ingestion.consume()          # pull the log
            batch = self.processor.process(records)     # -> ProcessedBatch

        # ---------- L5: storage ----------
        with self.tracer.span("L5.persist"):
            receipt = self.storage.persist(run_id, records, batch)

        # ---------- L6: MCP agent access (read path) ----------
        with self.tracer.span("L6.agent_tool_call"):
            agent_response = self.gateway.call_tool(
                spiffe_id="spiffe://agents/clinical-copilot",
                token="Bearer demo",
                tool="summarize_trends",
                prompt="Summarize the last hour of vitals trends.",
                result=f"{receipt.tsdb_points} points; "
                       f"{len(batch.alarms)} alarms; MRN:12345 seen.",
            )

        # ---------- L7: batch training from the coreset ----------
        with self.tracer.span("L7.train"):
            model = self.training.train(receipt.feature_keys)

        # ---------- L8: governance attestation ----------
        with self.tracer.span("L8.attest"):
            compliance = self.governance.attest(model=model)

        return BatchResult(
            receipt=receipt, batch=batch, agent_response=agent_response,
            model=model, compliance=compliance, trace=self.tracer.timeline(),
        )
