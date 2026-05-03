{{
  config(
    alias='daily_book_sales',
    materialized='table',
    tags=['bookstore', 'analytics']
  )
}}

with cleaned_orders as (
    select * from {{ ref('stg_orders') }}
    where status = 'paid'
),

inventory as (
    select * from {{ source('warehouse', 'book_catalog') }}
    where stock_count > 0
),

featured_titles as (
    select isbn, promo_label
    from {{ ref('stg_promotions') }}
    where active_from <= current_date
)

select
    o.book_isbn,
    i.title,
    i.author,
    coalesce(f.promo_label, 'standard') as promo_label,
    sum(o.copies_sold) as copies_sold,
    sum(o.revenue) as revenue
from cleaned_orders o
join inventory i on i.isbn = o.book_isbn
left join featured_titles f on f.isbn = o.book_isbn
where o.order_date >= '{{ var("window_start") }}'
group by 1, 2, 3, 4
