# Public data schema (structure only)

No records or row counts are included. Sensitive fields are retained only as field names so that the code interface is auditable.

## `comment_labels`

| field | type |
|---|---|
| `comment_id` | `TEXT` |
| `evaluation_object` | `TEXT` |
| `evidence_basis` | `TEXT` |
| `stance` | `TEXT` |
| `ability_type` | `TEXT` |
| `ability_property` | `TEXT` |

## `comments_anon`

| field | type |
|---|---|
| `comment_id` | `TEXT` |
| `note_id` | `TEXT` |
| `root_comment_id` | `TEXT` |
| `parent_comment_id` | `TEXT` |
| `comment_level` | `INTEGER` |
| `author_hash` | `TEXT` |
| `target_user_hash` | `TEXT` |
| `content` | `TEXT` |
| `create_ts_ms` | `INTEGER` |
| `create_time_cn` | `TEXT` |
| `like_count_raw` | `TEXT` |
| `like_count_num` | `REAL` |
| `sub_comment_count_raw` | `TEXT` |
| `sub_comment_count_num` | `REAL` |
| `ip_location` | `TEXT` |
| `show_tags_json` | `TEXT` |
| `first_seen_at` | `TEXT` |
| `last_seen_at` | `TEXT` |
| `deleted_inferred` | `INTEGER` |
| `raw_path` | `TEXT` |
| `schema_version` | `TEXT` |
| `content_raw` | `TEXT` |
| `content_clean` | `TEXT` |
| `semantic_eligible` | `INTEGER` |
| `nonverbal_only` | `INTEGER` |
| `short_text_flag` | `INTEGER` |
| `text_length` | `INTEGER` |
| `source_type` | `TEXT` |
| `relation_eligible` | `INTEGER` |
| `author_hash_anon` | `TEXT` |
| `target_user_hash_anon` | `TEXT` |
| `is_note_author` | `INTEGER` |
| `author_identifiable` | `INTEGER` |
| `is_orphan` | `INTEGER` |

## `crawl_runs`

| field | type |
|---|---|
| `crawl_run_id` | `TEXT` |
| `task_type` | `TEXT` |
| `query_id` | `TEXT` |
| `sort_type` | `TEXT` |
| `started_at` | `TEXT` |
| `finished_at` | `TEXT` |
| `status` | `TEXT` |
| `crawler_version` | `TEXT` |
| `upstream_commit` | `TEXT` |
| `request_count` | `INTEGER` |
| `success_count` | `INTEGER` |
| `failure_count` | `INTEGER` |
| `raw_manifest_path` | `TEXT` |

## `note_metric_snapshots`

| field | type |
|---|---|
| `snapshot_id` | `TEXT` |
| `note_id` | `TEXT` |
| `captured_at` | `TEXT` |
| `liked_count_raw` | `TEXT` |
| `liked_count_num` | `REAL` |
| `collected_count_raw` | `TEXT` |
| `collected_count_num` | `REAL` |
| `comment_count_raw` | `TEXT` |
| `comment_count_num` | `REAL` |
| `share_count_raw` | `TEXT` |
| `share_count_num` | `REAL` |

## `note_tags`

| field | type |
|---|---|
| `note_id` | `TEXT` |
| `tag_id` | `TEXT` |
| `tag_name` | `TEXT` |
| `tag_type` | `TEXT` |

## `notes_anon`

| field | type |
|---|---|
| `note_id` | `TEXT` |
| `note_type` | `TEXT` |
| `title` | `TEXT` |
| `description` | `TEXT` |
| `author_hash` | `TEXT` |
| `publish_ts_ms` | `REAL` |
| `publish_time_cn` | `TEXT` |
| `last_update_ts_ms` | `REAL` |
| `first_seen_at` | `TEXT` |
| `last_seen_at` | `TEXT` |
| `detail_status` | `TEXT` |
| `latest_raw_path` | `TEXT` |
| `schema_version` | `TEXT` |
| `author_recovered` | `TEXT` |
| `author_confidence` | `TEXT` |
| `description_synthetic` | `TEXT` |
| `has_original_desc` | `INTEGER` |
| `has_any_desc` | `INTEGER` |
| `source_type` | `TEXT` |
| `author_hash_anon` | `TEXT` |

## `search_hits`

| field | type |
|---|---|
| `hit_id` | `TEXT` |
| `crawl_run_id` | `TEXT` |
| `query_id` | `TEXT` |
| `note_id` | `TEXT` |
| `model_type` | `TEXT` |
| `result_page` | `REAL` |
| `rank_in_page` | `REAL` |
| `global_rank` | `REAL` |
| `xsec_token` | `TEXT` |
| `captured_at` | `TEXT` |
| `raw_path` | `TEXT` |

## `search_queries`

| field | type |
|---|---|
| `query_id` | `TEXT` |
| `query_text` | `TEXT` |
| `query_group` | `TEXT` |
| `priority` | `INTEGER` |
| `enabled` | `INTEGER` |
| `created_at` | `TEXT` |

## `v_comment_base`

| field | type |
|---|---|
| `comment_id` | `TEXT` |
| `note_id` | `TEXT` |
| `root_comment_id` | `TEXT` |
| `parent_comment_id` | `TEXT` |
| `comment_level` | `INTEGER` |
| `author_hash_secure` | `TEXT` |
| `target_user_hash_secure` | `TEXT` |
| `content_raw` | `TEXT` |
| `content_clean` | `TEXT` |
| `create_ts_ms` | `INTEGER` |
| `like_count_num` | `REAL` |
| `sub_comment_count_num` | `REAL` |
| `schema_version` | `TEXT` |
| `source_type` | `TEXT` |
| `is_note_author` | `INTEGER` |
| `author_identifiable` | `INTEGER` |
| `is_orphan` | `INTEGER` |
| `semantic_eligible` | `INTEGER` |
| `nonverbal_only` | `INTEGER` |
| `short_text_flag` | `INTEGER` |
| `relation_eligible` | `INTEGER` |

## `v_comment_context`

| field | type |
|---|---|
| `comment_id` | `TEXT` |
| `note_id` | `TEXT` |
| `note_title` | `TEXT` |
| `note_description_available` | `` |
| `parent_comment_text` | `TEXT` |
| `current_comment_text` | `TEXT` |
| `comment_level` | `INTEGER` |
| `is_note_author` | `INTEGER` |
| `schema_version` | `TEXT` |

## `v_comment_labeled`

| field | type |
|---|---|
| `comment_id` | `TEXT` |
| `note_id` | `TEXT` |
| `root_comment_id` | `TEXT` |
| `parent_comment_id` | `TEXT` |
| `comment_level` | `INTEGER` |
| `author_hash` | `TEXT` |
| `target_user_hash` | `TEXT` |
| `content` | `TEXT` |
| `create_ts_ms` | `INTEGER` |
| `create_time_cn` | `TEXT` |
| `like_count_raw` | `TEXT` |
| `like_count_num` | `REAL` |
| `sub_comment_count_raw` | `TEXT` |
| `sub_comment_count_num` | `REAL` |
| `ip_location` | `TEXT` |
| `show_tags_json` | `TEXT` |
| `first_seen_at` | `TEXT` |
| `last_seen_at` | `TEXT` |
| `deleted_inferred` | `INTEGER` |
| `raw_path` | `TEXT` |
| `schema_version` | `TEXT` |
| `content_raw` | `TEXT` |
| `content_clean` | `TEXT` |
| `semantic_eligible` | `INTEGER` |
| `nonverbal_only` | `INTEGER` |
| `short_text_flag` | `INTEGER` |
| `text_length` | `INTEGER` |
| `source_type` | `TEXT` |
| `relation_eligible` | `INTEGER` |
| `author_hash_anon` | `TEXT` |
| `target_user_hash_anon` | `TEXT` |
| `is_note_author` | `INTEGER` |
| `author_identifiable` | `INTEGER` |
| `is_orphan` | `INTEGER` |
| `evaluation_object` | `TEXT` |
| `evidence_basis` | `TEXT` |
| `stance` | `TEXT` |
| `ability_type` | `TEXT` |
| `ability_property` | `TEXT` |

## `v_corpus_source`

| field | type |
|---|---|
| `source_type` | `TEXT` |
| `comments` | `` |
| `notes` | `` |
| `users` | `` |
| `semantic_eligible` | `` |

## `v_note_base`

| field | type |
|---|---|
| `note_id` | `TEXT` |
| `title` | `TEXT` |
| `description` | `TEXT` |
| `tags` | `` |
| `note_type` | `TEXT` |
| `author_hash_secure` | `TEXT` |
| `publish_ts_ms` | `REAL` |
| `detail_status` | `TEXT` |
| `schema_version` | `TEXT` |
| `source_type` | `TEXT` |
| `liked_count_num` | `REAL` |
| `collected_count_num` | `REAL` |
| `comment_count_num` | `REAL` |
| `share_count_num` | `REAL` |
| `query_count` | `` |
| `query_groups` | `` |
| `best_rank` | `` |

## `v_note_metrics`

| field | type |
|---|---|
| `note_id` | `TEXT` |
| `total_comments` | `` |
| `root_comments` | `` |
| `reply_comments` | `` |
| `author_comments` | `` |
| `avg_comment_likes` | `` |
| `max_comment_likes` | `` |
| `unique_commenters` | `` |

## `v_reply_pairs`

| field | type |
|---|---|
| `note_id` | `TEXT` |
| `root_comment_id` | `TEXT` |
| `source_comment_id` | `TEXT` |
| `reply_comment_id` | `TEXT` |
| `source_user` | `TEXT` |
| `reply_user` | `TEXT` |
| `reply_is_note_author` | `INTEGER` |
| `source_text` | `TEXT` |
| `reply_text` | `TEXT` |
| `source_like_count` | `` |
| `reply_like_count` | `` |
| `thread_reply_count` | `` |
| `has_parent_marker` | `` |

## `v_search_lineage`

| field | type |
|---|---|
| `query_id` | `TEXT` |
| `query_text` | `TEXT` |
| `query_group` | `TEXT` |
| `crawl_run_id` | `TEXT` |
| `sort_type` | `` |
| `note_id` | `TEXT` |
| `result_page` | `REAL` |
| `rank_in_page` | `REAL` |
| `global_rank` | `REAL` |
| `captured_at` | `TEXT` |

## `v_user_participation`

| field | type |
|---|---|
| `user_hash_secure` | `TEXT` |
| `root_comment_count` | `` |
| `reply_count` | `` |
| `note_count` | `` |
| `first_seen` | `` |
| `last_seen` | `` |
