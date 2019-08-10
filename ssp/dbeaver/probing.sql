select count(domain_userid) from analytics.events;
select distinct event_name from analytics.events;

select count(event_name) as n, event_name from analytics.events group by event_name order by n desc;

select count(event_name) as n, event_name from analytics.events where (partner_key = 'Partner A') group by event_name order by n desc;

select count(event_name) as n, event_name from analytics.events where (partner_key = 'Partner B') group by event_name order by n desc;

select collector_tstamp as t, product_id as pid, event_name as en, dvce_screenwidth as vw from analytics.events where (partner_key = 'Partner A') and (dvce_screenwidth > 320) and not (event_name = 'viewed_product') order by vw, pid, t, en desc limit 1000;

select collector_tstamp as t, product_id as pid, product_domain as pd, event_name as en, dvce_screenwidth as vw from analytics.events where (partner_key = 'Partner A') and (dvce_screenwidth > 320) and not (event_name = 'viewed_product') order by vw, pid, t, en desc limit 1000;

select product_domain as pd, count(*) from analytics.events where (partner_key = 'Partner A') group by pd order by pd;
select product_domain as pd, count(*) from analytics.events where (partner_key = 'Partner A') and (event_name = 'selected_category') group by pd order by pd;

select product_domain as pd, count(*) from analytics.events where (partner_key = 'Partner B') group by pd order by pd;
select product_domain as pd, count(*) from analytics.events where (partner_key = 'Partner B') and (event_name = 'selected_category') group by pd order by pd;

select partner_key, count(*) from analytics.events where (product_domain is null) group by partner_key;

select partner_key, event_name, count(*) from analytics.events where (product_domain is null) group by partner_key, event_name order by partner_key, event_name;

select partner_key, event_name, count(*) from analytics.events where (product_domain is not null) group by partner_key, event_name order by partner_key, event_name;

select partner_key, ab_slot1_variant, count(*) from analytics.events where (product_domain is null) group by partner_key, ab_slot1_variant order by partner_key, ab_slot1_variant;
select partner_key, ab_slot1_variant, count(*) from analytics.events where (product_domain is not null) group by partner_key, ab_slot1_variant order by partner_key, ab_slot1_variant;

select product_domain, partner_key, count(event_name) as n from analytics.events group by product_domain, partner_key order by product_domain, partner_key;

select event_name, count(event_name) as n from analytics.events where (partner_key = 'Partner A') and (product_domain LIKE '%') group by event_name;


select prediction_size, count(*) from analytics.events where (partner_key = 'Partner A') and (product_domain = 'Dresses') and (event_name = 'completed_profiling') group by prediction_size;


select 
	event_name, 
	sum((partner_key = 'Partner A')::int) as "A", 
	sum((partner_key = 'Partner B')::int) as "B"
from analytics.events
group by event_name;


select
	ab_slot1_variant, count(ab_slot1_variant)
from analytics.events
group by ab_slot1_variant;

select 
	event_name, 
	sum((product_domain is null)::int) as "NULL", 
	sum((product_domain is not null)::int) as "Non-NULL" 
from analytics.events 
where 
	(partner_key = 'Partner A')
group by event_name;


select 
	event_name, 
	sum((ab_slot1_variant = 'Control')::int) as "Control", 
	sum((ab_slot1_variant = 'Test')::int) as "Test" 
from analytics.events 
where 
	(partner_key = 'Partner A') and
	((event_name = 'viewed_product') or (event_name = 'ordered_variant'))
group by event_name;

