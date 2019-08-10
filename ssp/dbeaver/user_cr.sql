
-- RA, 2019-08-05

drop table if exists temp_BaseTable;
--
create temporary table temp_BaseTable as
(
	select
		domain_userid,
		event_name,
		ab_slot1_variant,
		product_id,
		product_domain,
		collector_tstamp as tstamp
	from
		analytics.events
	where
		(partner_key ilike 'Partner A')
		and
		(event_name in ('viewed_product', 'ordered_variant'))
);


--select
--	domain_userid,
--	event_name,
--	ab_slot1_variant as slot1,
--	product_id as pid,
--	max(product_id) over (partition by domain_userid order by (case when (product_id is null) then '1970-01-01' else tstamp end) asc rows unbounded preceding) as product_id,
--	tstamp
--from
--	temp_BaseTable
--where 
--	(domain_userid = '0011fe66206025d17b3a724959fa7f3d1c396586')
--order by
--	tstamp
--;
	

--select *
--from temp_BaseTable
--where (domain_userid = '0011fe66206025d17b3a724959fa7f3d1c396586')
--order by tstamp
--limit 100;

select
	domain_userid, ab_slot1_variant,
	count(null or (event_name = 'viewed_product')) as viewed_product,
	count(null or (event_name = 'ordered_variant')) as ordered_variant
from
	analytics.events
where
	(partner_key ilike 'Partner A')
group by
	1, 2
;
