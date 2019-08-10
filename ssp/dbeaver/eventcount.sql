
-- RA, 2019-07-20


-- Event count per user histogram
with EventCount as (
	select 
		partner_key, ab_slot1_variant, domain_userid, count(event_name) as event_count
	from
		analytics.events
	group by
		partner_key, ab_slot1_variant, domain_userid
)
select 
	partner_key, ab_slot1_variant, event_count, count(domain_userid) as freq
from 
	EventCount
group by
	partner_key, ab_slot1_variant, event_count
order by
	partner_key, ab_slot1_variant, event_count
;
