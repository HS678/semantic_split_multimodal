from MSL.protocol import DATASET_PROTOCOLS
from MSL.datasets.mhealth import MHEALTH_FOLD_SUBJECTS
from MSL.datasets.pamap2 import PAMAP2_EVALUATION_SUBJECTS
from MSL.datasets.uci_har import UCI_HAR_TEST_SUBJECTS, UCI_HAR_TRAIN_SUBJECTS
from MSL.datasets.iemocap import IEMOCAP_FOLD_SESSIONS
from experiments.common import formal_folds


def assert_disjoint(train_groups, test_groups):
    assert set(int(value) for value in train_groups).isdisjoint(int(value) for value in test_groups)


# 验证 protocol、runner 和 loader 对 fold count 的定义一致。
def test_fold_counts_match_protocol_and_runners():
    assert DATASET_PROTOCOLS["uci_har"]["fold_count"] is None
    assert formal_folds("uci_har") == [None]
    assert DATASET_PROTOCOLS["mhealth"]["fold_count"] == len(MHEALTH_FOLD_SUBJECTS) == 5
    assert formal_folds("mhealth") == [1, 2, 3, 4, 5]
    assert DATASET_PROTOCOLS["pamap2"]["fold_count"] == len(PAMAP2_EVALUATION_SUBJECTS) == 8
    assert formal_folds("pamap2") == list(range(1, 9))
    assert DATASET_PROTOCOLS["iemocap"]["fold_count"] == len(IEMOCAP_FOLD_SESSIONS) == 5
    assert formal_folds("iemocap") == [1, 2, 3, 4, 5]


# 验证 UCI-HAR 使用官方 subject-disjoint split。
def test_uci_har_official_subject_split_is_disjoint():
    assert_disjoint(UCI_HAR_TRAIN_SUBJECTS, UCI_HAR_TEST_SUBJECTS)
    assert len(UCI_HAR_TRAIN_SUBJECTS) == 21
    assert len(UCI_HAR_TEST_SUBJECTS) == 9


# 验证 MHEALTH 每折 train/test subject 无交集。
def test_mhealth_group_leakage_protocol():
    covered = set()
    for fold, (train_subjects, test_subjects) in MHEALTH_FOLD_SUBJECTS.items():
        assert 1 <= int(fold) <= 5
        assert len(train_subjects) == 8
        assert len(test_subjects) == 2
        assert_disjoint(train_subjects, test_subjects)
        covered.update(int(value) for value in test_subjects)
    assert covered == set(range(1, 11))


# 验证 PAMAP2 为 8-fold LOSO，且 subject 109 不参与正式 evaluation folds。
def test_pamap2_loso_group_leakage_protocol():
    assert PAMAP2_EVALUATION_SUBJECTS == [101, 102, 103, 104, 105, 106, 107, 108]
    for fold, test_subject in enumerate(PAMAP2_EVALUATION_SUBJECTS, start=1):
        train_subjects = [value for value in PAMAP2_EVALUATION_SUBJECTS if value != test_subject]
        assert 1 <= fold <= 8
        assert len(train_subjects) == 7
        assert_disjoint(train_subjects, [test_subject])


# 验证 IEMOCAP 为 5-fold session LOSO。
def test_iemocap_session_leakage_protocol():
    covered = set()
    for fold, (train_sessions, test_sessions) in IEMOCAP_FOLD_SESSIONS.items():
        assert 1 <= int(fold) <= 5
        assert len(train_sessions) == 4
        assert len(test_sessions) == 1
        assert_disjoint(train_sessions, test_sessions)
        covered.update(int(value) for value in test_sessions)
    assert covered == {1, 2, 3, 4, 5}


# 验证 subject/session 前缀 sample ids 在 group-disjoint 协议下天然 disjoint。
def test_group_based_sample_id_scheme_is_disjoint():
    train_ids = {f"subject{subject}:window0" for subject in [1, 2, 3]}
    test_ids = {f"subject{subject}:window0" for subject in [4, 5]}
    assert train_ids.isdisjoint(test_ids)
