
-- RA, 20190720


select
	domain_userid, product_id, collector_tstamp, event_name
from
	analytics.events
where
	(domain_userid in ('932f02de2c0b57889ac52be8d7d5af46da38aac0'))
order by
	domain_userid,
	collector_tstamp;


-- "Average user" click-through graph

with
--
BaseTable as
(
	select 
		domain_userid, product_id,
		collector_tstamp,
		--(case when (event_name = 'completed_profiling') then (collector_tstamp + (interval '2 seconds')) else collector_tstamp end) as collector_tstamp, 
		event_name,
		(
			case 
				when (event_name = 'viewed_product') then 10
				when (event_name = 'opened_editor') then 21
				when (event_name = 'opened_brand_list') then 22
				when (event_name = 'selected_brand') then 23
				when (event_name = 'selected_category') then 24
				when (event_name = 'selected_size') then 25
				when (event_name = 'completed_profiling') then 26
				when (event_name = 'added_variant_to_cart') then 30
				when (event_name = 'ordered_variant') then 40
			else
				null
			end
		) as event_rank
	from
		analytics.events
	where
		(partner_key = 'Partner A')
		and
		(product_id is not null)
		and
		(ab_slot1_variant = 'Test')
),
-- Event ranks
ER as (
	select
		domain_userid, 
		product_id,
		collector_tstamp, event_name,
		(rank() over(partition by domain_userid order by collector_tstamp, event_rank)) as seq,
		(count(collector_tstamp) over(partition by domain_userid)) as tot
	from
		BaseTable
)
/** -- TEST:
select * from ER order by domain_userid, seq limit 1000;
-- END TEST **/
,
-- Full event chain for each user
FEC as (
	select 
		ER1.domain_userid, 
		ER1.event_name as event1,
		ER2.event_name as event2,
		not (ER1.product_id = ER2.product_id) as product_id_changed,
		(1.0 / ER1.tot) as rel_freq
	from
		ER as ER1
	left join 
		-- Note: left join creates event2 = [null] as an "exit"-event indicator
		ER as ER2
	on
		(ER1.domain_userid = ER2.domain_userid) 
		and
		-- Two consecutive events
		(ER1.seq + 1 = ER2.seq)
),
-- Event chain histogram for each user
EC as (
	select 
		domain_userid, 
		event1, event2, product_id_changed, sum(rel_freq) as rel_freq
	from
		FEC
	group by 
		domain_userid, 
		event1, event2, product_id_changed
)

/** -- TEST:
select
	*
from EC
where 
	(event1 = 'opened_editor') 
	and 
	(event2 = 'selected_size')
	and
	(not product_id_changed)
	and 
	(rel_freq > 0)
order by
	rel_freq desc, domain_userid
limit 20;
-- END TEST **/

-- Event chain histogram for the "average user"
select 
	event1, event2, product_id_changed, 
	-- Some users may not have the combination of event1, event2
	(sum(rel_freq) / (select count(distinct domain_userid) from EC)) as rel_freq
from 
	EC
group by
	event1, event2, product_id_changed
order by
	event1, event2, product_id_changed
limit 222;

