"""Build the 22 October supervisor package from verified aggregate evidence.

Copies only explicit metadata/config/evidence allowlists, never row predictions,
selected indices, train/dev features or tokens. B-E results are not assumed to
exist. The report states outstanding audits, approvals and unimplemented
ablations; it is a progress report rather than an automatic GO declaration.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from skq_exp.artifacts import atomic_json, sha256_file
from skq_exp.config import ExperimentConfig


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers), *["| " + " | ".join(str(c) for c in r) + " |" for r in rows]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",default="results/Bao_cao_2026_10-22")
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    output=(project/args.output).resolve()
    evidence=output/"01_bang_chung"
    data=project/"data/phase_features_v4/runtime_dev_only"
    audit=json.loads((evidence/"audit_v4_train_dev.json").read_text(encoding="utf-8"))
    endpoints=json.loads((evidence/"fulltrain_ab_endpoints.json").read_text(encoding="utf-8"))
    headroot=project/"results/cq_v4/analysis_headroom_all_20261022"
    head=json.loads((headroot/"headroom_decision.json").read_text())
    ci=json.loads((headroot/"family_bootstrap_2000_macro_f1.json").read_text())
    for p in (output/"02_du_lieu_metadata", output/"03_cau_hinh", output/"04_ma_nguon_va_kiem_thu",output/"05_archive_ky_thuat"):
        p.mkdir(parents=True,exist_ok=True)
    for name in ("DATA_CARD_V4.md","COURSE_FAMILY_SENSITIVITY.md","splits.json","bundle_manifest.json","formula_policy.json","preprocessing.json","selector_feature_columns.json","learner_feature_columns.json","validation.json"):
        shutil.copy2(data/name,output/"02_du_lieu_metadata"/name)
    for name in ("headroom_decision.json","family_bootstrap_2000_macro_f1.json"):
        shutil.copy2(headroot/name,evidence/name)
    configs=sorted((project/"configs").glob("cq_v4_m0_lr_tune_*.json"))+[project/"configs/cq_v4_panel_dev_rho005_seeds11_29_47_v1.json",project/"configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json"]
    for config in configs:
        ExperimentConfig.from_json(config)
        shutil.copy2(config,output/"03_cau_hinh"/config.name)
    shutil.copy2(project/"docs/cq_v4/panel_execution_plan_review1022.json",output/"03_cau_hinh/panel_execution_plan_review1022.json")
    for name in ("compute_cq_v4_dev_endpoints.py","audit_cq_v4_dev.py","prepare_cq_v4_review_package.py"):
        shutil.copy2(project/"scripts"/name,output/"04_ma_nguon_va_kiem_thu"/name)
    shutil.copy2(project/"tests/test_cq_v4_evidence_preparation.py",output/"04_ma_nguon_va_kiem_thu/test_cq_v4_evidence_preparation.py")
    archive_names=("skq_cq_v4_qd16_preflight_g10_2026_10_22.tar.gz","skq_cq_v4_qd16_group_a_controls_2026_10_22.tar.gz")
    for name in archive_names:
        shutil.copy2(project/"results/cq_v4"/name,output/"05_archive_ky_thuat"/name)
    resource_path=project/"results/cq_v4"/archive_names[1]
    with tarfile.open(resource_path) as archive:
        resource=json.load(archive.extractfile("artifacts/cq_v4_qd16_group_a_controls/resource_decision_all.json"))
    atomic_json(evidence/"qd16_group_a_summary.json",resource)
    checkrows=[(c["check"],c["status"],str(c["detail"])) for c in audit["checks"]]
    (output/"AUDIT_V4_VA_GIOI_HAN.md").write_text(
        "# Audit train dev CQ V4\n\n"+table(["Kiểm tra thực thi","Kết quả","Phạm vi"],checkrows)+
        "\n\n12 kiểm tra cục bộ trên đây không đồng nghĩa đóng toàn bộ kiểm tra 1–12 của review. Còn đối chiếu sự kiện nguồn/cutoff, human review 20 trường hợp, nguồn công thức và các contract selector/QP trên panel. Fit preprocessing chỉ được kiểm theo manifest, chưa tái dựng độc lập. Không đọc feature/nhãn test.\n",encoding="utf-8")
    frows=[]
    for run in endpoints["runs"]:
        m=run["mean_metrics"]
        frows.append([run["method_id"],*[f"{m[k]:.6f}" for k in ("f1_macro","mcc","worst_class_recall","pr_auc_macro","log_loss","ece_15","ul_course_equal")],f"{run['pcc']['ec_mean_p1_p4']:.6f}"])
    byphase=[]
    labels=[]
    for run in endpoints["runs"]:
        for phase,m in run["overall_by_phase"].items():
            pcc=run["pcc"]["per_phase"][phase]
            byphase.append([run["method_id"],phase,f"{m['f1_macro']:.6f}",f"{m['mcc']:.6f}",f"{pcc['ec_mae_pcc']:.6f}",f"{pcc['signed_course_error']:+.6f}",pcc["eligible_courses"]])
        for phase,rows in run["per_label_by_phase"].items():
            for row in rows:
                fmt=lambda value: "NA" if value is None else f"{value:.6f}"
                labels.append([run["method_id"],phase,["E","G","A"][int(row["label"])],row["support"],*[fmt(row.get(k)) for k in ("precision","recall","f1","pr_auc","roc_auc")]])
    overall_rows=[]
    for metric in endpoints["runs"][0]["mean_metrics"]:
        overall_rows.append([metric,*[f"{run['mean_metrics'][metric]:.6f}" for run in endpoints["runs"]]])
    (output/"KET_QUA_FULLTRAIN_CHI_TIET.md").write_text(
        "# Kết quả FullTrain A B CQ V4\n\nXGBoost natural; một model mỗi thiết kế, dùng lại cho bốn dev phase. Dev P4 được dùng early stopping nên đây là development evidence. Nhãn E=0, G=1, A=2.\n\n"+
        table(["Thiết kế","Macro-F1","MCC","Worst recall","Macro PR-AUC","Log loss","ECE","UL đều theo khóa","EC PCC"],frows)+
        "\n\n"+table(["Thiết kế","Phase","Macro-F1","MCC","EC PCC","Sai lệch có dấu","Khóa n≥30"],byphase)+
        "\n\nEC được tính theo course_id, bootstrap/split theo course_family_id. Toàn bộ accuracy, precision/recall/F1 macro/micro/weighted, kappa, ROC/PR-AUC, Brier, calibration, confusion và từng lớp có trong JSON `01_bang_chung/fulltrain_ab_endpoints.json`.\n\nA có Macro-F1/MCC và EC tốt hơn B ở cấu hình này; B giữ recall E tốt hơn A. Chưa suy ra superiority trên test. A8 gần như không đổi vì effective train prior và prior đích giống nhau trong thiết kế natural FullTrain; điều này không loại bỏ vai trò A8 cho selector khác.\n\n"+
        table(["Thiết kế","Phase","Lớp","Support","Precision","Recall","F1","PR-AUC","ROC-AUC"],labels)+
        "\n\n## Toàn bộ metric trung bình P1–P4\n\n"+
        table(["Metric",*[run["method_id"] for run in endpoints["runs"]]],overall_rows)+
        "\n\n## Ma trận nhầm lẫn\n\nHàng là nhãn thật E/G/A; cột là dự đoán E/G/A.\n\n"+
        "\n\n".join(f"### {run['method_id']} {phase}\n\n"+table(["Thật / dự đoán","E","G","A"],[[["E","G","A"][i],*row] for i,row in enumerate(matrix)]) for run in endpoints["runs"] for phase,matrix in run["confusion_matrices"].items()),encoding="utf-8")
    hrows=[]
    for ratio,v in head["ratios"].items():
        values=np.asarray(list(v["seed_values"].values()))
        hrows.append([f"{float(ratio)*100:g}%",len(values),f"{values.mean():.6f}",f"{values.std(ddof=1):.6f}",f"{values.std(ddof=1)/np.sqrt(len(values)):.6f}",f"{v['delta_full_minus_c02']:+.6f}",v["eligible"]])
    rrows=[]
    for method,item in resource["methods"].items():
        fit=item["fit"]
        rrows.append([method,len(fit["observations"]),f"{fit['predicted_seconds']:.6f}",f"{fit['upper_prediction_seconds']:.6f}",f"{fit['upper_prediction_peak_rss_bytes']/2**30:.3f}",item["decision"]["decision"]])
    probes=audit["probes"]
    prows=[]
    for name in ("majority","shuffle_labels_lr","shuffle_labels_xgb","missing_mask_lr"):
        phases=probes[name].get("per_phase",probes[name])
        for phase in ("P1","P2","P3","P4"):
            if phase in phases:
                m=phases[phase]
                prows.append([name,phase,f"{m['f1_macro']:.6f}",f"{m['mcc']:.6f}",str(m.get("roc_auc_macro"))])
    statusrows=[
        ["V4/hash/split/cohort","Đã kiểm tra train/dev và nhận data card","Đối chiếu nhãn/cutoff nguồn và cô duyệt deviation","PARTIAL"],
        ["FullTrain A/B XGB","8/8 dev evaluations; endpoint recomputed, IDs verified","LR và sensitivity balanced còn chưa chạy trên V4","XGB DONE"],
        ["Headroom","4 rho × 5 seed × 4 phase; rho*=0,5%; CI 2.000 family-bootstrap","Cô duyệt rho*/tham số; CI đều chứa 0","DONE DEV / PENDING APPROVAL"],
        ["G10","10/10 baseline toy PASS; 36/36 targeted tests trong archive","Không thay thế benchmark lớn","DONE TECHNICAL"],
        ["QĐ16","A: 4 phương pháp × 9 phép đo","B–E; workload replacement800k; native N/A; overhead/lịch GĐ2","PARTIAL"],
        ["Probes","Majority toàn dev; ID-only, shuffle LR/XGB, missing-mask LR capped20k","Probes capped không thay kiểm định leakage đầy đủ","DIAGNOSTIC DONE"],
        ["LR Y21","8 config V4 đúng grid; standardized train-only; natural/balanced","Chưa fit8 config; chưa chọn LR hội tụ","PREPARED"],
        ["Panel dev","13 runnable IDs ×3 seed; native CoreTab N/A ghi riêng","QD16 ba candidate và phê duyệt M0 trước khi chạy","PREPARED"],
        ["Ablation","A8 endpoint đã có; C03 matching có option","Winner; A2 class-only và A4 no-QP chưa có runtime switch","PARTIAL"],
        ["Repo/tài liệu","Code/test/config và bộ evidence riêng","PR/merge/tag, references/novelty rà soát đủ yêu cầu","PARTIAL"],
        ["M1/test","Giữ test khóa","Quyết định QĐ5 và freeze do cô duyệt 08/11","LOCKED / PENDING"],
    ]
    report="""# Báo cáo tiến độ CQ V4 theo Review ngày 22 tháng 10 năm 2026

Kính gửi cô, nhóm đã nhận CQ V4 chia theo họ khóa học, chạy FullTrain-A/B và headroom trên dev, xác minh baseline fidelity và hoàn tất nhóm A của hiệu chuẩn tài nguyên. Báo cáo này tổng hợp bằng chứng hiện có và các việc còn thiếu để cô xem xét điểm kiểm tra M0-V4 ngày 29/10, tiếp tục công việc train/dev đến M1 ngày 08/11. Chưa có kết quả test và chưa tự kết luận GO.

QĐ18 hiện có hiệu lực: NO-GO ICCIES 2027; tiếp tục nhánh train/dev đến M1. Nơi công bố và quyết định freeze do giảng viên duyệt. Không suy từ việc notebook success sang đáp ứng mọi tiêu chí nghiên cứu.

## 1. Đối chiếu yêu cầu và tiến độ

"""+table(["Nhóm yêu cầu","Đã làm được","Chưa đủ/chờ","Trạng thái trung thực"],statusrows)
    report+="\n\n## 2. Kết quả FullTrain và endpoint hai cấp\n\n"+table(["Thiết kế","Macro-F1","MCC","Worst recall","Macro PR-AUC","Log loss","ECE","UL đều theo khóa","EC PCC"],frows)
    report+="\n\nMỗi dev phase có 22.717 dòng, 179 course, 173 family; 67 course có n≥30. Pool B có 527.624 snapshot, FullTrain-A có 131.906 lượt ghi danh cùng cohort train. EC thấp hơn là tốt hơn; signed deviation được báo chi tiết theo phase. Kết quả là một model XGB trên mỗi thiết kế, không coi bốn phase là bốn lần lặp độc lập.\n"
    report+="\n## 3. Headroom QĐ12/QĐ21\n\n"+table(["rho","Seed","C02 mean","SD","SE","FullB−C02","Gap≥0,01"],hrows)
    report+="\n\nFullTrain-B UL=0,429165; rho*=0,5% là mức lớn nhất thỏa gap0,01, không dùng fallback. 5% là mức mô tả phụ có C02 UL cao nhất (0,471448). CI family-bootstrap 2.000 lần đều chứa0; không khẳng định superiority. File JSON đi kèm lưu đầy đủ CI và half-width.\n"
    report+="\n## 4. QĐ16 và fidelity\n\n"+table(["Selector","Số đo","Dự báo giây","Cận trên giây","Cận trên RSS GiB","Quyết định"],rrows)
    report+="\n\nCả36 phép đo success, đúng budget. Môi trường T4×2, max_threads=4; các selector nhómA chủ yếu CPU. Cỡ800k là workload lấy mẫu phase×class có hoàn lại từ train527.624: khai báo riêng và xin cô chấp thuận; không coi là800k dòng độc lập. RAM là RSS process-tree tuyệt đối, gồm nền runtime. Cận trên là diagnostic từ log-log, chưa là bảo đảm thống kê thời gian. Chưa có overhead đủ cho lịch dev4ngày/test3ngày. CoreTab native binary-only cần xử lý N/A được duyệt; không thay thế âm thầm bằng OVA. Trần3giờ toàn lệnh calibration không tương đương timeout một lần chọn. Moderate-DS chưa xuất passes_over_train.\n"
    report+="\n## 5. Audit và probes cục bộ\n\n"+table(["Probe","Phase","Macro-F1","MCC","Macro ROC-AUC"],prows)
    report+="\n\nCác probe học dùng mẫu train20.000, seed20261022, đánh giá toàn dev từng phase; ID-only báo ở P4. Missing-mask đọc từ raw train/dev trước impute, không tái dựng từ zero. Label-shuffle gần majority là chẩn đoán có ích nhưng không chứng minh không leakage. Mask có tín hiệu dự đoán cần được báo cùng hạn chế. Audit có12 kiểm tra cục bộ PASS; chưa đóng toàn bộ kiểm tra1–12 của review.\n"
    report+="\n## 6. Các xác nhận xin cô duyệt\n\n1. V4 family split, hash và khác biệt công thức/feature so với công bố nguồn; nguồn dữ liệu vẫn cần xác nhận label/event audit.\n2. rho*=0,5%, nmin=30 và deltaC=0,01 để khóa trước panel.\n3. Workload800k có hoàn lại do trainV4<800k; hoặc chỉ định cỡ thay thế rồi đo lại nhất quán.\n4. Ledger CoreTab native N/A trên multiclass, CoreTab OVA adapted là baseline chạy thực.\n5. Các failure/timeout giữ đúng phạm vi đo; Kernel Thinning Compress++ sau shard theo QĐ22.\n\n## 7. Công việc đến M1\n\nHoàn tất B–E và dự báo lịch; bổ sung audit/provenance; chạy lưới8 LR; hoàn thiện PR/tag/references/novelty và xin duyệt M0. Sau duyệt chạy13 method runnable×3seed (39 selection/fit blocks, tối đa156 dev evaluations), native N/A ghi riêng. C02 seed11/29/47 chỉ được reuse nếu config/data/method artifact identity khớp; không tự chép metric từ headroom. Chọn winner và chạy C03 matched,A2,A4,A8; phân tích UL/EC/per-class/phase, family CI và Rbreak. Ngày08/11 trình QĐ5 và freeze; test ở GĐ2 sau duyệt. Không có cam kết thắng baseline hoặc accept.\n\nNếu candidate không vượt baseline, QĐ5(c) cho phép định vị benchmark/đánh đổi endpoint và chi phí. Độ rộngCI hiện còn lớn phải được giữ trong kết luận.\n"
    report+="\n\n## 8. CI và lưu ý giao thức bổ sung\n\n"+table(["rho","CI 95%: FullB−C02","Nửa độ rộng"],[[name,f"[{row['ci95_lower']:.6f}; {row['ci95_upper']:.6f}]",f"{row['ci95_half_width']:.6f}"] for name,row in ci["ratios"].items()])
    ids=probes["id_only_lr"]
    report+=f"\n\nID-only P4: course_id Macro-F1={ids['course_id']['macro_f1']:.6f}; teacher+school Macro-F1={ids['teacher_school']['macro_f1']:.6f}. Đây là diagnostic capped, không thay learner chính.\n"
    report+="\nData card báo temporal overlap 924/1.219 family train (75,7998%), vượt 5%; đối chứng NOLFO cho FullTrain-B và winner cần được khai báo ở GĐ2/luận văn. Chưa chạy NOLFO, giữ split chính. Data card ghi bỏ đối chiếu TEMPO-DQ theo chỉ dẫn người dùng, trong khi O24 cần cô xác nhận deviation/quan hệ nguồn; cần làm rõ khác biệt này, không tự coi yêu cầu của cô đã đóng.\n"
    regression=output/"04_ma_nguon_va_kiem_thu/pytest_regression.xml"
    if regression.is_file():
        import xml.etree.ElementTree as ET
        suite=ET.parse(regression).getroot().find("testsuite")
        report+=f"\nKiểm thử mã chuẩn bị: {suite.attrib['tests']} tests; failures={suite.attrib['failures']}, errors={suite.attrib['errors']}, skipped={suite.attrib['skipped']}. XML kèm trong thư mục04. Kiểm thử mã và audit dữ liệu thật là hai loại evidence riêng.\n"
    for old,new in {"fit8":"fit 8", "replacement800k":"replacement 800k", "capped20k":"capped 20k", "gap0,01":"gap 0,01", "chứa0":"chứa 0", "Cả36":"Cả 36", "nhómA":"nhóm A", "Cỡ800k":"Cỡ 800k", "train527.624":"train 527.624", "là800k":"là 800k", "dev4ngày/test3ngày":"dev 4 ngày / test 3 ngày", "Trần3giờ":"Trần 3 giờ", "train20.000":"train 20.000", "seed20261022":"seed 20261022", "có12":"có 12", "kiểm tra1–12":"kiểm tra 1–12", "Workload800k":"Workload 800k", "trainV4<800k":"train V4 < 800k", "lưới8":"lưới 8", "chạy13":"chạy 13", "runnable×3seed":"runnable × 3 seed", "đa156":"đa 156", "seed11/29/47":"seed 11/29/47", "matched,A2,A4,A8":"matched, A2, A4, A8", "Ngày08/11":"Ngày 08/11", "rộngCI":"rộng CI"}.items():
        report=report.replace(old,new)
    (output/"BAO_CAO_TIEN_DO_CQ_V4_2026_10_22.md").write_text(report,encoding="utf-8")
    instructions="""# Chuẩn bị chạy LR và panel CQ V4

Tám config LR ở `03_cau_hinh` đã validate, CHƯA CHẠY. Main natural, balanced chỉ sensitivity. Dùng CPU; solver lbfgs/newton-cholesky, C=1/0,1, standardized train-only. Grid đúng tối đa8 config, không tune thêm bằng test.

Trong checkout repo cùng phiên bản code:

```text
python -m skq_exp.cli prepare-external --config configs/cq_v4_m0_lr_tune_01_natural_lbfgs_c1_v1.json --input-dir <runtime_dev_only> --dataset course_quality_v4
python -m skq_exp.cli run --config configs/<config_lr>.json --resume
python scripts/compute_cq_v4_dev_endpoints.py --input-dir <runtime_dev_only> --artifacts artifacts --experiment-id <experiment_id_lr> --output artifacts/experiments/<experiment_id_lr>/cq_dev_endpoints.json
```

Sau khi chạy đủ8, dùng `scripts/select_cq_v1_lr_config.py` với8 option `--config`. Tên script cũ nhưng đọc config/endpoint V4 được; chỉ rank fit hội tụ và đủ4phase. Nếu tất cả không hội tụ giữ LR descriptive. Chưa có cell Kaggle LR mới trong bộ này; sẽ gửi cell trực tiếp trong chat khi bắt đầu chạy. FullTrain-A LR chỉ chạy nếu cô yêu cầu; Y21 ưu tiên tune FullTrain-B.

Panel config ở `03_cau_hinh/cq_v4_panel_dev_rho005_seeds11_29_47_v1.json` là PREPARED, chưa có duyệt M0 nên chưa launch. Sau duyệt và QĐ16, chia job bằng `--method` và `--selector-seed` của CLI. Không chạy nhiều selector đồng thời trong cùng runtime đo timing. Sửa method options/budget/seed làm thay identity; không reuse tùy ý. Early stopping XGB dùng devP4 như hiện tại.

A2 và A4 mới có đặc tả trong plan JSON, chưa có runtime switch; không gọi cấu hình đó là runnable. A8 tính từ prediction; C03 cùng allocation/floor với winner. Không tự chọn winner khi chưa có đủ panel.
"""
    (output/"CHUAN_BI_LR_VA_PANEL.md").write_text(instructions,encoding="utf-8")
    (output/"README_GUI_GIANG_VIEN.md").write_text("""# Danh sách tài liệu gửi giảng viên

Gửi trước `BAO_CAO_TIEN_DO_CQ_V4_2026_10_22.md`. Hai phụ lục `KET_QUA_FULLTRAIN_CHI_TIET.md` và `AUDIT_V4_VA_GIOI_HAN.md` giải thích số liệu/giới hạn. Kèm `01_bang_chung`, `02_du_lieu_metadata`, `03_cau_hinh`, `04_ma_nguon_va_kiem_thu`, `05_archive_ky_thuat` và `SHA256SUMS.txt` nếu gửi nguyên bộ.

Không chứa token, train/dev features, raw predictions hay selected indices. Split metadata có course/family assignment; chỉ gửi trực tiếp theo thỏa thuận dữ liệu. Hai archive kỹ thuật kèm theo không có prediction cấp dòng. Archive FullTrain gốc (có prediction cấp dòng) giữ riêng, báo hash trong inventory, cung cấp khi cô cần bằng kênh được phép. Bốn archive headroom gốc cũng giữ ngoài gói. JSON aggregate đủ đối chiếu bảng hiện tại.

Repo: https://github.com/UIT23521830/skq-experiment ; nhánh cq-v1-protocol-fixes. Commit từng thí nghiệm có trong manifest, không dùng HEAD mới thay commit lịch sử. Chưa merge PR/tag của lô chuẩn bị này. Notebook URLs/version do người chạy bổ sung; không tự đoán URL. B–E, LR và panel chưa có kết quả trong bản đóng gói này.
""",encoding="utf-8")
    archives=[project/"results/cq_v4/skq_cq_v4_fulltrain_ab_dev_2026_10_22.tar.gz",*[project/"results/cq_v4"/n for n in archive_names],*sorted((project/"results/cq_v4/rho").glob("*.tar.gz"))]
    atomic_json(evidence/"source_archive_inventory.json",{"archives":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p),"original_contains_row_predictions": "fulltrain" in p.name or "headroom" in p.name} for p in archives],"test_data_read":False})
    atomic_json(evidence/"preparation_status.json",{
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=project,text=True).strip(),
        "test_data_read":False,"lr_grid":"VALIDATED_NOT_EXECUTED", "panel":"VALIDATED_PENDING_QD16_AND_APPROVAL", "A2_A4":"SPECIFICATION_ONLY_IMPLEMENTATION_PENDING",
        "publication_or_go_status":"NOT_GRANTED_BY_THIS_PACKAGE",
    })
    paths=sorted(p for p in output.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS.txt","GOI_GUI_CO.zip"})
    (output/"SHA256SUMS.txt").write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}" for p in paths)+"\n",encoding="utf-8")
    with zipfile.ZipFile(output/"GOI_GUI_CO.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*paths,output/"SHA256SUMS.txt"]:
            archive.write(path,path.relative_to(output).as_posix())
    print(json.dumps({"output":str(output),"files":len(paths)+1,"test_data_read":False}))


if __name__=="__main__":
    main()
