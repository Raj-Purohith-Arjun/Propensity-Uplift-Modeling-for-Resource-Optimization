"""
SQL query templates for data extraction.
Designed for use with operational databases at Applied Materials scale.
"""

CUSTOMER_FEATURES_QUERY = """
WITH base_customers AS (
    SELECT
        c.customer_id,
        c.age,
        DATEDIFF(CURRENT_DATE, c.acquisition_date) / 30.0  AS tenure_months,
        c.region,
        c.segment,
        c.is_active
    FROM customers c
    WHERE c.is_active = 1
      AND c.acquisition_date <= DATEADD(month, -3, CURRENT_DATE)
),
interaction_stats AS (
    SELECT
        i.customer_id,
        COUNT(CASE WHEN i.interaction_date >= DATEADD(day, -30, CURRENT_DATE)
                   THEN 1 END)                               AS num_interactions_30d,
        COUNT(CASE WHEN i.interaction_date >= DATEADD(day, -90, CURRENT_DATE)
                   THEN 1 END)                               AS num_interactions_90d,
        AVG(CASE WHEN i.interaction_date >= DATEADD(day, -30, CURRENT_DATE)
                 THEN i.spend_amount END)                    AS avg_spend_30d,
        AVG(CASE WHEN i.interaction_date >= DATEADD(day, -90, CURRENT_DATE)
                 THEN i.spend_amount END)                    AS avg_spend_90d,
        MIN(DATEDIFF(CURRENT_DATE, i.interaction_date))      AS recency_days
    FROM interactions i
    WHERE i.interaction_date >= DATEADD(day, -180, CURRENT_DATE)
    GROUP BY i.customer_id
),
product_stats AS (
    SELECT
        p.customer_id,
        COUNT(DISTINCT p.product_id)                         AS num_products_owned,
        COUNT(DISTINCT p.product_category) * 1.0
            / NULLIF(COUNT(DISTINCT p.product_id), 0)        AS product_diversity_score,
        SUM(CASE WHEN p.is_cross_sell = 1 THEN 1 ELSE 0 END) * 1.0
            / NULLIF(COUNT(*), 0)                            AS cross_sell_ratio
    FROM products_owned p
    GROUP BY p.customer_id
),
support_stats AS (
    SELECT
        s.customer_id,
        COUNT(CASE WHEN s.ticket_date >= DATEADD(day, -90, CURRENT_DATE)
                   THEN 1 END)                               AS support_tickets_90d,
        AVG(CASE WHEN s.ticket_date >= DATEADD(day, -90, CURRENT_DATE)
                 THEN s.is_escalated END)                    AS escalation_rate,
        AVG(CASE WHEN s.ticket_date >= DATEADD(day, -90, CURRENT_DATE)
                 THEN s.resolution_hours END)                AS resolution_time_avg
    FROM support_tickets s
    GROUP BY s.customer_id
),
rfm AS (
    SELECT
        customer_id,
        NTILE(5) OVER (ORDER BY recency_days DESC)           AS frequency_score,
        NTILE(5) OVER (ORDER BY num_interactions_90d)        AS recency_score,
        NTILE(5) OVER (ORDER BY avg_spend_90d)               AS monetary_score
    FROM interaction_stats
)
SELECT
    bc.customer_id,
    bc.age,
    bc.tenure_months,
    bc.region,
    bc.segment,
    COALESCE(ist.num_interactions_30d, 0)  AS num_interactions_30d,
    COALESCE(ist.num_interactions_90d, 0)  AS num_interactions_90d,
    COALESCE(ist.avg_spend_30d, 0)         AS avg_spend_30d,
    COALESCE(ist.avg_spend_90d, 0)         AS avg_spend_90d,
    COALESCE(ist.recency_days, 999)        AS recency_days,
    COALESCE(rfm.frequency_score, 1)       AS frequency_score,
    COALESCE(rfm.monetary_score, 1)        AS monetary_score,
    COALESCE(ps.num_products_owned, 0)     AS num_products_owned,
    COALESCE(ps.product_diversity_score,0) AS product_diversity_score,
    COALESCE(ps.cross_sell_ratio, 0)       AS cross_sell_ratio,
    COALESCE(ss.support_tickets_90d, 0)   AS support_tickets_90d,
    COALESCE(ss.escalation_rate, 0)       AS escalation_rate,
    COALESCE(ss.resolution_time_avg, 0)   AS resolution_time_avg
FROM base_customers bc
LEFT JOIN interaction_stats  ist ON bc.customer_id = ist.customer_id
LEFT JOIN product_stats       ps ON bc.customer_id = ps.customer_id
LEFT JOIN support_stats       ss ON bc.customer_id = ss.customer_id
LEFT JOIN rfm                    ON bc.customer_id = rfm.customer_id
"""

AB_TEST_RESULTS_QUERY = """
SELECT
    e.customer_id,
    e.treatment_flag                          AS treatment,
    e.experiment_id,
    e.assignment_date,
    COALESCE(o.converted, 0)                  AS converted,
    COALESCE(o.revenue_amount, 0)             AS revenue_amount,
    COALESCE(o.days_to_conversion, NULL)      AS days_to_conversion
FROM experiment_assignments e
LEFT JOIN outcomes o
    ON e.customer_id = o.customer_id
   AND o.outcome_date BETWEEN e.assignment_date
                          AND DATEADD(day, 30, e.assignment_date)
WHERE e.experiment_id = :experiment_id
  AND e.assignment_date BETWEEN :start_date AND :end_date
"""

REGION_ENCODING_QUERY = """
SELECT region, ROW_NUMBER() OVER (ORDER BY region) - 1 AS region_encoded
FROM (SELECT DISTINCT region FROM customers) sub
"""

SEGMENT_ENCODING_QUERY = """
SELECT segment, ROW_NUMBER() OVER (ORDER BY segment) - 1 AS segment_encoded
FROM (SELECT DISTINCT segment FROM customers) sub
"""
