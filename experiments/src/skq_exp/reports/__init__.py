"""Package này gom artifact đã có thành bảng dài và bảng so sánh.

Nó không chạy lại selector hay model. Nhờ vậy việc chỉnh cách trình bày không làm
thay đổi prediction và mỗi số trong báo cáo vẫn truy ngược được về run_id.
"""

from .aggregate import aggregate_results
from .evidence import audit_claim_ledger, create_claim_record, create_evidence_manifest

__all__ = [
    "aggregate_results",
    "audit_claim_ledger",
    "create_claim_record",
    "create_evidence_manifest",
]

