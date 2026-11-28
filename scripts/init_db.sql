-- 初始化 PostgreSQL 库表
-- 用法：psql -U postgres -d aq_platform -f scripts/init_db.sql
-- 说明：observation 按年 RANGE 分区（2021–2024）；其余表为普通表。
--       本脚本幂等，可重复执行（IF NOT EXISTS / ON CONFLICT DO NOTHING）。

BEGIN;

-- 1. 数据集版本
CREATE TABLE IF NOT EXISTS dataset_version (
    dataset_version varchar(64)  PRIMARY KEY,
    name            varchar(128) NOT NULL,
    source          varchar(256) NOT NULL,
    time_start      timestamp    NOT NULL,
    time_end        timestamp    NOT NULL,
    station_count   smallint     NOT NULL,
    pollutant_list  text[]       NOT NULL,
    missing_rate    numeric(6,4),
    checksum        varchar(128),
    created_at      timestamp    NOT NULL DEFAULT now(),
    note            text
);

-- 2. 站点
CREATE TABLE IF NOT EXISTS station (
    station_id   varchar(32) PRIMARY KEY,
    name         varchar(64) NOT NULL,
    city         varchar(32),
    lon          numeric(9,6) NOT NULL,
    lat          numeric(9,6) NOT NULL,
    is_reference char(1) NOT NULL DEFAULT 'N' CHECK (is_reference IN ('Y', 'N'))
);

-- 3. 逐小时观测（按年分区）
CREATE TABLE IF NOT EXISTS observation (
    station_id varchar(32) NOT NULL REFERENCES station(station_id),
    ts         timestamp   NOT NULL,
    pm25 numeric(8,3),
    pm10 numeric(8,3),
    no2  numeric(8,3),
    o3   numeric(8,3),
    temp numeric(8,3),
    rh   numeric(8,3),
    ws   numeric(8,3),
    wd   numeric(8,3),
    pres numeric(8,3),
    PRIMARY KEY (station_id, ts)
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS observation_2021 PARTITION OF observation
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
CREATE TABLE IF NOT EXISTS observation_2022 PARTITION OF observation
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE IF NOT EXISTS observation_2023 PARTITION OF observation
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE IF NOT EXISTS observation_2024 PARTITION OF observation
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE INDEX IF NOT EXISTS idx_observation_ts ON observation (ts);

-- 4. 模型版本
CREATE TABLE IF NOT EXISTS model_version (
    model_id        varchar(64)  PRIMARY KEY,
    pollutant       varchar(16)  NOT NULL,
    model_type      varchar(32)  NOT NULL,
    weight_path     varchar(256) NOT NULL,
    dataset_version varchar(64)  NOT NULL REFERENCES dataset_version(dataset_version),
    hyperparams     jsonb        NOT NULL DEFAULT '{}'::jsonb,
    trained_at      timestamp    NOT NULL,
    note            text
);

-- 5. 实验记录
CREATE TABLE IF NOT EXISTS experiment_run (
    run_id          varchar(128) PRIMARY KEY,
    experiment_code varchar(8)   NOT NULL,
    model_id        varchar(64)  REFERENCES model_version(model_id),
    pollutant       varchar(16)  NOT NULL,
    seed            integer      NOT NULL,
    dataset_version varchar(64)  REFERENCES dataset_version(dataset_version),
    config          jsonb        NOT NULL,
    metrics         jsonb        NOT NULL,
    best_epoch      smallint,
    duration_s      integer,
    status          varchar(16)  NOT NULL DEFAULT 'success',
    started_at      timestamp,
    finished_at     timestamp,
    note            text
);

CREATE INDEX IF NOT EXISTS idx_experiment_model ON experiment_run (model_id);
CREATE INDEX IF NOT EXISTS idx_experiment_code  ON experiment_run (experiment_code);
CREATE INDEX IF NOT EXISTS idx_experiment_poll  ON experiment_run (pollutant, seed);

-- 6. 关联网络结果
CREATE TABLE IF NOT EXISTS network_result (
    network_id      varchar(64) PRIMARY KEY,
    pollutant       varchar(16) NOT NULL,
    builder         varchar(32) NOT NULL,
    window_from     timestamp   NOT NULL,
    window_to       timestamp   NOT NULL,
    params          jsonb       NOT NULL,
    dataset_version varchar(64) REFERENCES dataset_version(dataset_version),
    edge_count      integer     NOT NULL,
    fdr_pass        integer,
    modularity      numeric(8,5),
    metrics         jsonb,
    storage_path    varchar(256),
    created_at      timestamp   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_network_pollutant ON network_result (pollutant, builder);

-- 7. 异步任务
CREATE TABLE IF NOT EXISTS task (
    task_key    varchar(64) PRIMARY KEY,
    type        varchar(32) NOT NULL,
    params      jsonb       NOT NULL,
    status      varchar(16) NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'success', 'failed')),
    progress    smallint    NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    result_ref  varchar(128),
    error       text,
    created_at  timestamp   NOT NULL DEFAULT now(),
    finished_at timestamp
);

CREATE INDEX IF NOT EXISTS idx_task_status ON task (status);
CREATE INDEX IF NOT EXISTS idx_task_type   ON task (type);

-- 8. 种子数据：数据集版本（数值取自 data/processed 实测）
INSERT INTO dataset_version (
    dataset_version, name, source, time_start, time_end,
    station_count, pollutant_list, missing_rate, note
) VALUES (
    'ds-2021-2024-v1',
    '北京国控 24 站逐小时空气质量',
    'quotsoft china_sites_YYYYMMDD.csv + ERA5-Land',
    '2021-01-01', '2024-12-31 23:00',
    24, ARRAY['pm25', 'pm10', 'no2', 'o3'], 0.1155,
    '张量 35064x24x4，总体缺失率 11.55%；3671A 长时段停摆，评估按时段 mask'
) ON CONFLICT (dataset_version) DO NOTHING;

-- 9. 种子数据：24 个国控站（与 data/raw/stations.csv 一致）
INSERT INTO station (station_id, name, city, lon, lat, is_reference) VALUES
    ('1001A', '万寿西宫',     '北京', 116.3621, 39.8784, 'N'),
    ('1002A', '定陵(对照点)', '北京', 116.2202, 40.2915, 'Y'),
    ('1003A', '东四',         '北京', 116.4174, 39.9289, 'N'),
    ('1004A', '天坛',         '北京', 116.4072, 39.8863, 'N'),
    ('1005A', '农展馆',       '北京', 116.4620, 39.9365, 'N'),
    ('1006A', '官园',         '北京', 116.3392, 39.9295, 'N'),
    ('1007A', '海淀万柳',     '北京', 116.2878, 39.9611, 'N'),
    ('1008A', '顺义新城',     '北京', 116.6636, 40.1350, 'N'),
    ('1009A', '怀柔镇',       '北京', 116.6275, 40.3275, 'N'),
    ('1010A', '昌平镇',       '北京', 116.2296, 40.2170, 'N'),
    ('1011A', '奥体中心',     '北京', 116.3966, 39.9821, 'N'),
    ('1012A', '古城',         '北京', 116.1840, 39.9136, 'N'),
    ('3281A', '延庆夏都',     '北京', 115.9718, 40.4529, 'N'),
    ('3417A', '平谷新城',     '北京', 117.0709, 40.1465, 'N'),
    ('3418A', '密云新城',     '北京', 116.8336, 40.4095, 'N'),
    ('3671A', '门头沟三家店', '北京', 116.0880, 39.9726, 'N'),
    ('3672A', '丰台云岗',     '北京', 116.1541, 39.8204, 'N'),
    ('3673A', '通州东关',     '北京', 116.6804, 39.9081, 'N'),
    ('3674A', '房山燕山',     '北京', 115.9577, 39.7603, 'N'),
    ('3675A', '大兴旧宫',     '北京', 116.4672, 39.7733, 'N'),
    ('3694A', '延庆石河营',   '北京', 115.9860, 40.4563, 'N'),
    ('3695A', '怀柔新城',     '北京', 116.6151, 40.3053, 'N'),
    ('3696A', '丰台小屯',     '北京', 116.2545, 39.8768, 'N'),
    ('3697A', '密云镇',       '北京', 116.8400, 40.3844, 'N')
ON CONFLICT (station_id) DO NOTHING;

COMMIT;
