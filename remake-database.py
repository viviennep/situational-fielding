import duckdb, pathlib
data_dir = pathlib.Path(__file__).resolve().parent / 'data'

con = duckdb.connect(data_dir / 'leaderboard.duckdb')

con.execute(f"""
    create or replace view all_plays as
    select *
    from read_parquet('{data_dir}/daily_data/*/*/*.parquet',
                      hive_partitioning=True);
""")

con.execute("""
    create or replace table leaderboard as
    with main_stats as (
        select 
            game_year as season,
            resp_fielder_name as fielder_name,
            count(*)          as plays,
            sum(visra)        as vioaa,
            sum(scsra)        as scoaa,
            sum(wpoe)         as wpoe,
            sum(wpoeli)       as wpoeli,
            min(
             date_diff('year',cast(resp_fielder_bday as date),make_date(game_year,7,1))
            ) as age,
            case
             when count(distinct fld_team)=1 then min(fld_team)
             else cast(count(distinct fld_team) as varchar) || 'TM'
            end as team
        from all_plays
        where is_of_play
        group by game_year,resp_fielder_name),
    pos_counts as (
        select
            game_year          as season,
            resp_fielder_name  as fielder_name,
            resp_fielder       as pos_code,
            count(*)           as cnt
        from all_plays
        where is_of_play
        group by game_year, resp_fielder_name, resp_fielder),
    primary_pos as (
        select
            season,
            fielder_name,
            pos_code as primary_pos_code
        from (
            select
                season,
                fielder_name,
                pos_code,
                cnt,
                row_number()
                    over(
                        partition by season, fielder_name
                        order by cnt desc, pos_code
                    ) as rn
            from pos_counts
        )
        where rn=1
    )
    select
        main_stats.*,
        primary_pos.primary_pos_code,
        case primary_pos.primary_pos_code
            when 2 then 'C'
            when 3 then '1B'
            when 4 then '2B'
            when 5 then '3B'
            when 6 then 'SS'
            when 7 then 'LF'
            when 8 then 'CF'
            when 9 then 'RF'
            else 'XX'
        end as primary_position
    from main_stats left join primary_pos using (season,fielder_name);
""")

con.close()
