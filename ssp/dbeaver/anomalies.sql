
-- RA, 20190719

/*
 * abbreviations:
 * 
 * e1 opened_editor
 * e2 opened_brand_list
 * e3 selected_brand
 * e4 selected_category
 * e5 selected_size
 * e6 completed_profiling
 * 
 */

/*
 * note:
 * 
 * sometimes selected_brand precedes opened_brand_list by a second
 * 
 */

/*
 * note:
 * 
 * select (null > 1);     -- null
 * select (null > null);  -- null
 * select null or true;   -- true
 * select null or false;  -- null
 * etc.
 * 
 */


select least(null);
-- null

select (false or (now() <= least(null, to_timestamp(0))));
-- null

select (now() >= '20180415');
-- true

select (now() <= '20180415');
-- false


-- Distinct user count
select partner_key, ab_slot1_variant, count(distinct domain_userid)
from analytics.events
group by partner_key, ab_slot1_variant;
--Partner A	Control	296724
--Partner A	Test	298061
--Partner B	[NULL]	593905


-- Data collection period
select min(collector_tstamp), max(collector_tstamp) 
from analytics.events;
--	2017-10-11 00:00:00
--	2018-05-16 07:00:00


-- Does product_id imply product_domain?
-- Only for Partner B 
select partner_key, product_id, string_agg(distinct product_domain, ', ')
from analytics.events
group by partner_key, product_id
having (count(distinct product_domain) > 1)
limit 111;
-- Partner B	Y0FUJ	Male Bottoms, Male Tops
-- Partner B	[NULL]	Dresses, Female Shoes, Female Tops, Male Bottoms, Male Shoes, Male Tops


-- All events
select string_agg(distinct event_name, ', ')
from analytics.events;
-- added_variant_to_cart, completed_profiling, opened_brand_list, opened_editor, ordered_variant, selected_brand, selected_category, selected_size, viewed_product


-- One suspicious case skipping events
select 
	event_name, domain_userid, product_domain, product_id, collector_tstamp
from 
	analytics.events
where 
	(domain_userid = 'e9ac9845986074e34ce17a84ad51e18cb2da7a82')
--	and
--	(product_domain = 'Female Bottoms')
order by
	collector_tstamp;


-- "Average user" click-through graph
-- Moved out



-- The main query I to find event anomalies (only the first pass)

with 
EventSequence(partner_key, domain_userid, product_domain, product_id, e1, e2, e3, e4, e5, e6)
as (
	select 
		partner_key, domain_userid, product_domain, product_id, 
		-- record the *first* event of the given type:
		min(case when (event_name = 'opened_editor')       then collector_tstamp else null end) as e1,
		min(case when (event_name = 'opened_brand_list')   then collector_tstamp else null end) as e2,
		min(case when (event_name = 'selected_brand')      then collector_tstamp else null end) as e3,
		min(case when (event_name = 'selected_category')   then collector_tstamp else null end) as e4,
		min(case when (event_name = 'selected_size')       then collector_tstamp else null end) as e5,
		min(case when (event_name = 'completed_profiling') then collector_tstamp else null end) as e6
	
	from analytics.events
	
--	where
--		(collector_tstamp >= '20180215')
	
--	where
--		(domain_userid = 'e9ac9845986074e34ce17a84ad51e18cb2da7a82')
	
	group by 
		partner_key, domain_userid, product_domain, product_id
)
select *
from EventSequence
where

	-- User started the size calculator
	-- First e1 = 'opened_editor'
	(e1 is not null)
	
	and 
	
	-- Anomaly in the events
	(
		-- Are there gaps in the events?
		(
			-- Try to find any gap
			
			-- First e1 = 'opened_editor'
			((e1 is null) and (least(e2, e3, e4, e5, e6) is not null))
			
			or
			
			-- First e2 = 'opened_brand_list'
			((e2 is null) and (least(e3, e4, e5, e6) is not null))
			
			or
			
			-- First e3 = 'selected_brand'
			((e3 is null) and (least(e4, e5, e6) is not null))
			
			or
			
			-- First e4 = 'selected_category'
			((e4 is null) and (least(e5, e6) is not null) and not (product_domain = 'Dresses'))
			
			or
			
			-- First e5 = 'selected_size'
			((e5 is null) and (least(e6) is not null))
		)
		
		or
		
		-- Is there an anomalous timing of events?
		not coalesce(
			-- Check if the timing is OK:
			(
				-- Allow for a one-second inaccuracy margin
				
				-- First e1 = 'opened_editor'
				((e1 is null) or (e1 <= coalesce((interval '1 second') + least(e2, e3, e4, e5, e6), e1)))
				
				and
				
				-- First e2 = 'opened_brand_list'
				((e2 is null) or (e2 <= coalesce((interval '1 second') + least(e3, e4, e5, e6), e2)))
				
				and
				
				-- First e3 = 'selected_brand'
				((e3 is null) or (e3 <= coalesce((interval '1 second') + least(e4, e5, e6), e3)))
				
				and
				
				-- First e4 = 'selected_category'
				-- No need to check (product_domain = 'Dresses')
				((e4 is null) or (e4 <= coalesce((interval '1 second') + least(e5, e6), e4)))
				
				and
				
				-- First e5 = 'selected_size'
				((e5 is null) or (e5 <= coalesce((interval '1 second') + least(e6), e5)))
				
				-- First e6 = 'completed_profiling
				-- May or may not happen
			),
			-- The above result shouldn't be 'null' logically, but just in case return 'false'
			false
		)
	)
	
order by
	partner_key, domain_userid, product_domain, product_id
	
limit 222;





-- The main query II to find event anomalies
-- Segment by event_group (opened_editor event initiates a new event_group)
-- Only for one Partner

with 
--
BaseTable as (
	select 
		domain_userid, product_domain, product_id, event_name, collector_tstamp
	from 
		analytics.events
	where
		(partner_key = 'Partner A')
		and
		(product_id is not null)
		and
		(event_name not in ('viewed_product', 'added_variant_to_cart', 'ordered_variant'))
),
--
OpenedEditor as (
	select
		domain_userid, product_id, collector_tstamp, 
		row_number() over(partition by domain_userid, product_id order by collector_tstamp) as opened_editor_no
	from
		BaseTable
	where
		(event_name = 'opened_editor')
),
--
Grouped as (
	select 
		BT.domain_userid, BT.product_domain, BT.product_id, BT.event_name, BT.collector_tstamp,
		max(OE.opened_editor_no) as event_group
	from BaseTable as BT
	join OpenedEditor as OE
	on 
		(BT.domain_userid = OE.domain_userid)
		and
		(BT.product_id = OE.product_id)
		and
		(BT.collector_tstamp >= OE.collector_tstamp)
	group by
		BT.domain_userid, BT.product_domain, BT.product_id, BT.event_name, BT.collector_tstamp
	order by
		BT.domain_userid, BT.collector_tstamp
),
--
EventSequence
as (
	select 
		domain_userid, product_domain, product_id, event_group,
		-- record the *first* event of the given type:
		min(case when (event_name = 'opened_editor')       then collector_tstamp else null end) as e1,
		min(case when (event_name = 'opened_brand_list')   then collector_tstamp else null end) as e2,
		min(case when (event_name = 'selected_brand')      then collector_tstamp else null end) as e3,
		min(case when (event_name = 'selected_category')   then collector_tstamp else null end) as e4,
		min(case when (event_name = 'selected_size')       then collector_tstamp else null end) as e5,
		min(case when (event_name = 'completed_profiling') then collector_tstamp else null end) as e6
	
	from 
		Grouped
	
	group by 
		domain_userid, product_domain, product_id, event_group
)
--
select *
from EventSequence
where

	-- User started the size calculator
	-- First e1 = 'opened_editor'
	(e1 is not null)
	
	and 
	
	-- Anomaly in the events
	(
		-- Are there gaps in the events?
		(
			-- Try to find any gap
			
			-- First e1 = 'opened_editor'
			((e1 is null) and (least(e2, e3, e4, e5, e6) is not null))
			
			or
			
			-- First e2 = 'opened_brand_list'
			((e2 is null) and (least(e3, e4, e5, e6) is not null))
			
			or
			
			-- First e3 = 'selected_brand'
			((e3 is null) and (least(e4, e5, e6) is not null))
			
			or
			
			-- First e4 = 'selected_category'
			((e4 is null) and (least(e5, e6) is not null) and not (product_domain = 'Dresses'))
			
			or
			
			-- First e5 = 'selected_size'
			((e5 is null) and (least(e6) is not null))
		)
		
		or
		
		-- Is there an anomalous timing of events?
		not coalesce(
			-- Check if the timing is OK:
			(
				-- Allow for an inaccuracy margin
				
				-- First e1 = 'opened_editor'
				((e1 is null) or (e1 <= coalesce((interval '2 seconds') + least(e2, e3, e4, e5, e6), e1)))
				
				and
				
				-- First e2 = 'opened_brand_list'
				((e2 is null) or (e2 <= coalesce((interval '2 seconds') + least(e3, e4, e5, e6), e2)))
				
				and
				
				-- First e3 = 'selected_brand'
				((e3 is null) or (e3 <= coalesce((interval '2 seconds') + least(e4, e5, e6), e3)))
				
				and
				
				-- First e4 = 'selected_category'
				-- No need to check (product_domain = 'Dresses')
				((e4 is null) or (e4 <= coalesce((interval '2 seconds') + least(e5, e6), e4)))
				
				and
				
				-- First e5 = 'selected_size'
				((e5 is null) or (e5 <= coalesce((interval '2 seconds') + least(e6), e5)))
				
				-- First e6 = 'completed_profiling
				-- May or may not happen
			),
			-- The above result shouldn't be 'null' logically, but just in case return 'false'
			false
		)
	)
	
order by
	domain_userid, product_domain, product_id, e1
	
limit 222;
