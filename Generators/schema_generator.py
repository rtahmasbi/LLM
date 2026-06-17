import rstr
import random
import re
from faker import Faker

fake = Faker()

# ── 1. UUID with underscores ───────────────────────────────────────────────
def gen_uuid_underscore():
    return rstr.xeger(r'[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}')

# ── 2. AWS SSO Role ────────────────────────────────────────────────────────
def gen_aws_sso_role():
    envs  = ['PROD', 'DEV', 'STAGING', 'QA']
    teams = ['DataTeam', 'PlatformTeam', 'SecurityTeam', 'MLTeam', 'OpsTeam']
    return f"AWSReservedSSO_{random.choice(envs)}_{random.choice(teams)}_{rstr.xeger(r'[0-9a-f]{16}')}"

# ── 3. Simple category label ───────────────────────────────────────────────
def gen_category_label():
    return random.choice(['dashboard', 'user', 'data_source', 'report', 'dataset', 'pipeline'])

# ── 4. Dashboard / assessment name ────────────────────────────────────────
def gen_dashboard_name():
    prefixes = ['KPI_Dashboard', 'Opp_Assessment', 'Revenue_Report', 'Usage_Metrics', 'Member_Summary']
    clients  = ['Humana', 'OakStreet', 'Bancorp', 'Stride', 'Cigna', 'Chime', 'Aetna']
    return f"{random.choice(prefixes)}___{random.choice(clients)}"

# ── 5. Email-style username ────────────────────────────────────────────────
def gen_email_username():
    domain = random.choice(['gmail_com', 'company_org', 'corp_net', 'yahoo_com', 'outlook_com', 'example_com', 'mail_com'])
    return f"{fake.first_name()[0].lower()}{fake.last_name().lower()}_{domain}"

# ── 6. AWS region ──────────────────────────────────────────────────────────
def gen_aws_region():
    return random.choice(['us_east_1', 'us_east_2', 'us_west_1', 'us_west_2',
                          'eu_west_1', 'eu_central_1', 'ap_southeast_1', 'ap_northeast_1'])

# ── 7. ALLCAPS field (no numeric suffix) ──────────────────────────────────
ALLCAPS_TOKENS = [
    'TOTAL', 'FIRST', 'LAST', 'IS', 'NUM', 'TICKET', 'LOGIN', 'COMPLIANCE',
    'BANK', 'STRIDE', 'BANCORP', 'TRANSFER', 'TRIAGE', 'STAGE', 'CLAIM',
    'AUTH', 'CALL', 'MEMBER', 'LOGIN', 'SUBMISSION', 'APPROVAL', 'REVIEW',
    'AUTHENTICATION', 'BROWSER', 'PASSWORD', 'OKTA', 'EXTERNAL', 'QUEUE',
    'AGENT', 'DATE', 'COUNT', 'FLAG', 'DAYS', 'STATUS', 'LIST', 'SLA_MET',
    'ATTEMPTS', 'TIMESTAMP', 'ID', 'EMAIL', 'NAME', 'TYPE', 'SCORE', 'TIME',
    'SLA', 'DURATION', 'START', 'END', 'STATE', 'RATE', 'INDEX', 'LEVEL',
]

def gen_allcaps_field():
    n = random.randint(2, 4)
    return '_'.join(random.choices(ALLCAPS_TOKENS, k=n))

# ── 8. ALLCAPS_FIELDNAME_NUMERICID  ← most common new pattern ─────────────
#   e.g. EMAIL_6954016, FIRST_NAME_6155087, SALESFORCE_LEAD_STATUS_1111111
FIELD_BASES = [
    'EMAIL', 'FIRST_NAME', 'LAST_NAME', 'COMPANY', 'TITLE', 'COUNTRY',
    'CITY', 'STATE', 'POSTAL_CODE', 'PHONE', 'WEBSITE', 'DOMAIN',
    'PERSONA', 'INDUSTRY', 'REGION', 'SFDC_ID', 'SFDC_TYPE', 'LEAD_ID',
    'ACCOUNT_ID', 'CONTACT_ID', 'OWNER_ID', 'PERSON_STATUS', 'IMPORT_SOURCE',
    'LEAD_SOURCE_C', 'LEAD_SOURCE_DETAIL_C', 'SEQUENCE_NAME', 'ASSIGNEE_C',
    'SALESFORCE_LEAD_STATUS', 'SALESFORCE_ACCOUNT_STATUS', 'SALESFORCE_LEAD_SOURCE',
    'SALESFORCE_DEMO_REQUEST_DATE', 'SALESFORCE_LAST_ACTIVITY_DATE',
    'SALESFORCE_NAMED_ACCOUNT_STATUS', 'SALESFORCE_SEQUENCE_NAME_FOR_AUTOMATION',
    'EMAIL_VALIDATION_STATUS', 'IS_EMAIL_VALID', 'IS_LEAD_SUPPRESSED',
    'IS_LEAD_PROCESSED', 'LEAD_UUID_FROM_SOURCE', 'LEAD_SALESFORCE_ID',
    'LEAD_SALESFORCE_OBJECT_TYPE', 'LEAD_LINKEDIN_URL', 'COMPANY_NAME',
    'COMPANY_SALESFORCE_ACCOUNT_ID', 'AUDIENCE_BATCH_UUID', 'AUDIENCE_CONFIG_UUID',
    'RED_ID', 'EE_SIZE', 'INTENT_SOURCE', 'STREET_ADDRESS', 'SNAPSHOT_UUID',
    'ROLE_ID', 'START_DATE', 'END_DATE', 'MANAGER_EMAIL', 'MANAGER_NAME',
    'EMPLOYMENT_TYPE', 'WORK_COUNTRY', 'WORK_STATE', 'ACCESS_TYPE',
    'STATIC_VALUE', 'SYNC_METADATA', 'RUN_DT', 'RECORD_ID',
    'PROGRAM', 'MAILBOX_ID', 'SEQUENCE_ID', 'ACTION_ID', 'ACTION_NAME',
    'EXPIRES_AT', 'FALLBACK_SEQUENCE_ID', 'INTENT_RECOMMENDATION_C',
]

def gen_field_with_id():
    base = random.choice(FIELD_BASES)
    # IDs cluster in certain ranges observed in data
    id_range = random.choice([
        (100000, 999999),
        (1000000, 9999999),
        (10000000, 99999999),
    ])
    numeric_id = random.randint(*id_range)
    return f"{base}_{numeric_id}"

# ── 9. static_value__NUMERICID  ────────────────────────────────────────────
def gen_static_value():
    numeric_id = random.randint(100000, 9999999)
    return f"static_value__{numeric_id}"

# ── 10. sync_metadata__NUMERICID ───────────────────────────────────────────
def gen_sync_metadata():
    numeric_id = random.randint(100000, 9999999)
    return f"sync_metadata__{numeric_id}"

# ── 11. NUMPREFIX_FIELDNAME_NUMERICID  ─────────────────────────────────────
#   e.g. 01_LAST_SYNCED_WITH_SFDC_640665, 09_FIRST_NAME_646484
def gen_numbered_field():
    num    = str(random.randint(1, 30)).zfill(2)
    base   = random.choice(FIELD_BASES)
    nid    = random.randint(100000, 999999)
    return f"{num}_{base}_{nid}"

# ── 12. client_vendor_fieldtype  (snake, no numeric suffix) ───────────────
#   e.g. bcnc_labcorp_roster_table_name, humana_labcorp_files_to_upload
CLIENTS  = ['bcnc', 'humana', 'ibx', 'conviva', 'moda', 'ssm', 'ckcc_detroit',
            'ckcc_gateway', 'ckcc_ipa', 'ckcc_nani_pc', 'ecp']
VENDORS  = ['labcorp', 'epic', 'mhc', 'navvis', 'hsx', 'adt', 'ibx', 'sftp']
CONFIG_KEYS = [
    'roster_table_name', 'select_statement', 'files_to_upload',
    'general_roster_config', 'roster_delimiter', 'encryption_recipient_email',
    'encryption_output_prefix', 'encryption_output_to_strive_bucket',
    'sftp', 'files_to_upload_contents', 'files_to_upload_encrypt',
    'files_to_upload_file_id', 'files_to_upload_output_filename',
    'files_to_upload_row_count', 'files_to_upload_use_start_timestamp',
    'files_to_upload_contents_timestamp_format',
    'files_to_upload_filename_timestamp_format',
    'member_engagement_sftp', 'adt_roster_table_name', 'adt_select_statement',
]

def gen_client_vendor_config():
    client = random.choice(CLIENTS)
    vendor = random.choice(VENDORS)
    key    = random.choice(CONFIG_KEYS)
    return f"{client}_{vendor}_{key}"

# ── 13. Survey question ────────────────────────────────────────────────────
SURVEY_TOPICS = [
    'NPS', 'EASE_OF_USE', 'FINANCIAL_GOALS', 'CHIME_VALUE', 'TRUST',
    'RELIABILITY', 'CHIME_CARES', 'LOGIN_EXP', 'ISSUE_RESOLUTION',
    'PRODUCT_KNOWLEDGE', 'CHIME_IMPROVEMENT', 'SOLUTIONS', 'PERSONALIZED_EMAILS',
]

def gen_survey_question():
    q_num  = random.randint(1, 25)
    topic  = random.choice(SURVEY_TOPICS)
    suffix = random.choice(['', '_NPS_GROUP', '_COMMENT', '_SCORE', ''])
    return f"Q{q_num}_{topic}{suffix}"

# ── 14. dotted table.column ────────────────────────────────────────────────
TABLE_NAMES = ['inspector_claims', 'inspector_disputes_table',
               'inspector_disputes_agg_metrics', 'inspector_users',
               'member_events', 'audit_log', 'call_records']
COL_NAMES   = ['claim_ext_id', 'member_ext_id', 'agent_email', 'created_at_date',
               'state', 'count', 'reason_code', 'dispute_type', 'assigned_date_date',
               'claim_age_in_days', 'first_last_name', 'supervisor_name']

def gen_dotted_table_col():
    return f"{random.choice(TABLE_NAMES)}.{random.choice(COL_NAMES)}"

# ── 15. VALUE_namespace_key ────────────────────────────────────────────────
VALUE_NS   = ['extra', 'payload', 'resource', 'table', 'normalizer_health_info', 'datadog']
VALUE_KEYS = ['created_by', 'schema', 'location', 'owner', 'table_type', 'num_rows',
              'num_bytes', 'is_partitioned', 'description', 'partition_keys',
              'view_query', 'last_altered', 'labels', 'retention_time', 'table_id']

def gen_value_nested():
    return f"VALUE_{random.choice(VALUE_NS)}_{random.choice(VALUE_KEYS)}"

# ── 16. Records_section_key ────────────────────────────────────────────────
RECORDS_SECTIONS = ['requestParameters', 'responseElements', 'userIdentity',
                    'tlsDetails', 'sessionContext']
RECORDS_KEYS     = ['bucket', 'functionName', 'roleArn', 'keyId', 'stackName',
                    'logGroupName', 'subnetId', 'vpcId', 'encryptionAlgorithm',
                    'accessKeyId', 'accountId', 'arn', 'principalId', 'type',
                    'cipherSuite', 'tlsVersion', 'mfaAuthenticated']

def gen_records_nested():
    return f"Records_{random.choice(RECORDS_SECTIONS)}_{random.choice(RECORDS_KEYS)}"

# ── 17. CloudTrail flat camelCase ──────────────────────────────────────────
CLOUDTRAIL_FLAT = [
    'eventName', 'eventSource', 'eventTime', 'eventType', 'eventVersion',
    'eventID', 'awsRegion', 'awsAccountId', 'sourceIPAddress', 'userAgent',
    'requestID', 'readOnly', 'managementEvent', 'recipientAccountId',
    'sharedEventID', 'errorCode', 'errorMessage', 'vpcEndpointId',
]

def gen_cloudtrail_flat():
    return random.choice(CLOUDTRAIL_FLAT)

# ── 18. CDK Resource nested ────────────────────────────────────────────────
CDK_RESOURCE_COMPONENTS = [
    ('operationalsyncdev', r'[0-9A-F]{8}', [
        'Properties_Engine', 'Properties_EngineVersion', 'Properties_DBInstanceClass',
        'Properties_AllocatedStorage', 'Properties_MultiAZ', 'Properties_StorageType',
        'Properties_StorageEncrypted', 'Properties_PubliclyAccessible',
        'Properties_Tags_Key', 'Properties_Tags_Value', 'Properties_VPCSecurityGroups',
        'Type', 'UpdateReplacePolicy', 'DeletionPolicy', 'Metadata_aws_cdk_path',
    ]),
    ('operationalsyncdevoperationalsyncdevproxy', r'[0-9A-F]{8}', [
        'Properties_Auth_AuthScheme', 'Properties_Auth_IAMAuth',
        'Properties_DBProxyName', 'Properties_EngineFamily', 'Properties_RequireTLS',
        'Properties_RoleArn', 'Properties_VpcSecurityGroupIds', 'Properties_VpcSubnetIds',
        'Properties_Tags_Key', 'Properties_Tags_Value', 'Type', 'Metadata_aws_cdk_path',
    ]),
    ('operationalsyncdevRDSProxySecurityGroup', r'[0-9A-F]{8}', [
        'Properties_GroupDescription', 'Properties_VpcId', 'Properties_CidrIp',
        'Properties_FromPort', 'Properties_ToPort', 'Properties_IpProtocol',
        'Properties_GroupId', 'Type', 'Metadata_aws_cdk_path',
    ]),
    ('operationalsyncdevS3ExportRole', r'[0-9A-F]{8}', [
        'Properties_AssumeRolePolicyDocument', 'Properties_Tags_Key',
        'Properties_Tags_Value', 'Properties_PolicyDocument',
        'Properties_PolicyName', 'Properties_Roles', 'Type', 'Metadata_aws_cdk_path',
    ]),
]

def gen_cdk_resource():
    prefix, hash_pat, props = random.choice(CDK_RESOURCE_COMPONENTS)
    hash_ = rstr.xeger(hash_pat)
    prop  = random.choice(props)
    return f"Resources_{prefix}{hash_}_{prop}"

# ── 19. WEB_N_CATEGORY_NUMERICID  ─────────────────────────────────────────
#   e.g. WEB_2_ACCOUNTING_870111, WEB_3_HRIS_407743
WEB_CATEGORIES = [
    'ACCOUNTING', 'BENEFITS_COMPENSATION', 'COMPLIANCE', 'DEI',
    'GLOBAL_WORKFORCE_MANAGEMENT', 'INTEGRATIONS_AND_3RD_PARTY_DATA',
    'IT_AND_TECHNOLOGY', 'ONBOARDING_AND_OFFBOARDING', 'PAYROLL_AND_TIME_TRACKING',
    'PEOPLE_AND_CULTURE', 'RECRUITING', 'REPORTING_AND_DATA_VIZ', 'SECURITY',
    'TALENT_MANAGEMENT', 'WORKFLOW_AUTOMATION', 'HRIS', 'IT_MGMT_APPS_DEVICES',
    'NEW_PRODUCTS_AND_FEATURES_BETA', 'PAYROLL_AND_T_AND_A', 'BENEFITS_AND_INSURANCE',
]

def gen_web_field():
    tier = random.randint(2, 3)
    cat  = random.choice(WEB_CATEGORIES)
    nid  = random.randint(100000, 999999)
    return f"WEB_{tier}_{cat}_{nid}"

# ── 20. g2Crowd field  ────────────────────────────────────────────────────
#   e.g. "g2Crowd company_size_1122384", "g2Crowd created_at_1122385"
G2_KEYS = ['company_size', 'created_at', 'document_title', 'message',
           'user_action', 'user_id', 'review_count', 'rating']

def gen_g2crowd_field():
    key = random.choice(G2_KEYS)
    nid = random.randint(100000, 9999999)
    return f"g2Crowd {key}_{nid}"

# ── 21. Mixed-case spaced (human-readable) ─────────────────────────────────
SPACED_FIELDS = ['Agent Email', 'Supervisor Email', '90 Days QA Score',
                 'Agent Name', 'Team Lead', 'QA Score', '30 Day Score',
                 'Handle Time', 'First Call Resolution']

def gen_spaced_field():
    return random.choice(SPACED_FIELDS)

# ── 22. snake_case HR/people ──────────────────────────────────────────────
def gen_snake_hr():
    return random.choice(['current_job_title', 'full_name', 'work_email',
                          'manager_email', 'manager_full_name', 'full_department_name',
                          'role_state', 'dd', 'mm', 'yyyy'])

# ── 23. lowercase_field_with_id  (e.g. city_1003216, email_1003211) ───────
LOWER_FIELD_BASES = ['email', 'first_name', 'last_name', 'city', 'country',
                     'phone', 'state', 'zip', 'static_value_']

def gen_lower_field_with_id():
    base = random.choice(LOWER_FIELD_BASES)
    nid  = random.randint(100000, 9999999)
    return f"{base}_{nid}"

# ── 24. SALESFORCE_FIELD_C_NUMERICID  ─────────────────────────────────────
#   e.g. ACCOUNT_STATUS_C_1122736, LEAD_SOURCE_C_1122000
SFDC_C_FIELDS = [
    'ACCOUNT_STATUS_C', 'ALIAS_TYPE_C', 'LEAD_SOURCE_C', 'LEAD_SOURCE_DETAIL_C',
    'PERSON_STATUS_SFDC_C', 'CHURN_RISK_LEVEL_C', 'RIPPLING_EMPLOYMENT_TYPE_C',
    'PASSWORD_MANAGEMENT_C', 'PRIMARY_CHURN_REASON_C', 'PRIMARY_CHURN_REASON_DETAIL_C',
    'PRIMARY_RISK_AREA_C', 'PERFORMANCE_MANAGEMENT_C', 'SERVICE_TERMINATION_DATE_C',
    'OUTSOURCED_MSP_COMPANY_NAME_C', 'SALARIED_EMPLOYEES_IN_UPCOMING_RUN_C',
    'ASSIGNED_USER_ROLE_C', 'ASSIGNEE_ALIAS_C', 'INTENT_RECOMMENDATION_C',
    'ALIAS_MAILBOX_ID_C', 'ACCOUNT_C', 'ASSIGNEE_C',
]

def gen_sfdc_c_field():
    base = random.choice(SFDC_C_FIELDS)
    nid  = random.randint(100000, 9999999)
    return f"{base}_{nid}"

# ── 25. BUSINESSCOM-style long field (Marketo/form field encoding) ─────────
#   e.g. BUSINESSCOMHRSOFTWAREDOYOUCURRENTLYUSEASOFTWARESOLUTION_1122376
BUSINESSCOM_FIELDS = [
    'BUSINESSCOMHRSOFTWAREDOYOUCURRENTLYUSEASOFTWARESOLUTION',
    'BUSINESSCOMHRSOFTWAREFUNCTIONS',
    'BUSINESSCOMHRWHATSOLUTIONSAREIMPORTANTTOEVALUATION',
    'BUSINESSCOMHRWHENDOYOUANTICIPATEADECISION',
    'BUSINESSCOMPAYROLLHOWAREYOUMANAGINGPAYROLL',
    'BUSINESSCOMPEOIMPORTANTSOLUTIONSFOREVALUATION',
    'BUSINESSCOMPEOWHATTYPEOFSERVICEAREYOUINTERESTEDIN',
]

def gen_businesscom_field():
    base = random.choice(BUSINESSCOM_FIELDS)
    nid  = random.randint(1000000, 9999999)
    return f"{base}_{nid}"



ALLCAPS_PREFIXES = ['TOTAL', 'FIRST', 'LAST', 'IS', 'NUM', 'TICKET', 'LOGIN',
                    'COMPLIANCE', 'BANK', 'STRIDE', 'BANCORP', 'TRANSFER',
                    'TRIAGE', 'STAGE', 'CLAIM', 'AUTH', 'CALL', 'MEMBER']
ALLCAPS_MIDS     = ['LOGIN', 'SUBMISSION', 'APPROVAL', 'REVIEW', 'AUTHENTICATION',
                    'BROWSER', 'PASSWORD', 'OKTA', 'EXTERNAL', 'INTERNAL',
                    'QUEUE', 'AGENT', 'TRANSFER', 'TICKET', 'SLA', 'DURATION']
ALLCAPS_SUFFIXES = ['DATE', 'COUNT', 'FLAG', 'DAYS', 'STATUS', 'LIST',
                    'SLA_MET', 'ATTEMPTS', 'TIMESTAMP', 'ID', 'EMAIL',
                    'NAME', 'TYPE', 'SCORE', 'RATE', 'TIME']

def gen_allcaps_field_v2(parts=None):
    n = parts or random.randint(2, 4)
    tokens = ([random.choice(ALLCAPS_PREFIXES)] +
              [random.choice(ALLCAPS_MIDS) for _ in range(n - 2)] +
              [random.choice(ALLCAPS_SUFFIXES)])
    return '_'.join(tokens)





# ══════════════════════════════════════════════════════════════════════════════
# Master generator registry
# ══════════════════════════════════════════════════════════════════════════════
GENERATORS = {
    'uuid_underscore'       : (gen_uuid_underscore,    0.5),
    'aws_sso_role'          : (gen_aws_sso_role,       0.5),
    'category_label'        : (gen_category_label,     0.5),
    'dashboard_name'        : (gen_dashboard_name,     0.5),
    'email_username'        : (gen_email_username,     0.5),
    'aws_region'            : (gen_aws_region,         0.5),
    'allcaps_field'         : (gen_allcaps_field,      2.0),   # common
    'gen_allcaps_field_v2'  : (gen_allcaps_field_v2,   2.0),   # common
    'field_with_id'         : (gen_field_with_id,      8.0),   # most common
    'static_value'          : (gen_static_value,       2.0),
    'sync_metadata'         : (gen_sync_metadata,      0.5),
    'numbered_field'        : (gen_numbered_field,     1.0),
    'client_vendor_config'  : (gen_client_vendor_config, 2.0),
    'survey_question'       : (gen_survey_question,    1.0),
    'dotted_table_col'      : (gen_dotted_table_col,   0.5),
    'value_nested'          : (gen_value_nested,       1.0),
    'records_nested'        : (gen_records_nested,     1.0),
    'cloudtrail_flat'       : (gen_cloudtrail_flat,    0.5),
    'cdk_resource'          : (gen_cdk_resource,       1.5),
    'web_field'             : (gen_web_field,          1.0),
    'g2crowd_field'         : (gen_g2crowd_field,      0.5),
    'spaced_field'          : (gen_spaced_field,       0.3),
    'snake_hr'              : (gen_snake_hr,           0.5),
    'lower_field_with_id'   : (gen_lower_field_with_id, 1.0),
    'sfdc_c_field'          : (gen_sfdc_c_field,       1.5),
    'businesscom_field'     : (gen_businesscom_field,  0.5),
}

def generate_column_names(n=100, weighted=True):
    """Generate n synthetic column_name values, optionally weighted by frequency."""
    names, weights = zip(*[(fn, w) for fn, w in GENERATORS.values()])
    fns = list(names)
    ws  = list(weights)
    
    chosen_fns = random.choices(fns, weights=ws, k=n) if weighted else \
                 [random.choice(fns) for _ in range(n)]
    return [fn() for fn in chosen_fns]

def generate_preview(n=3):
    print(f"{'PATTERN':<26} EXAMPLE")
    print("─" * 80)
    for name, (fn, w) in GENERATORS.items():
        for _ in range(n):
            print(f"{name:<26} {fn()}")
        print()

if __name__ == "__main__":
    generate_preview(n=5)

    print("\n── Weighted batch of 20 ──")
    for col in generate_column_names(100):
        print(col)
    
