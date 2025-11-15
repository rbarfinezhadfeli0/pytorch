# Documentation: `.github/scripts/trymerge.py`

## File Metadata

- **Path**: `.github/scripts/trymerge.py`
- **Size**: 91,286 bytes (89.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

# NB: the following functions are used in Meta-internal workflows
# (github_first_try_merge/my_handler.py) and thus have functionality limitations
# (no `git` command access, no network access besides the strict allow list):
#
# find_matching_merge_rule
# read_merge_rules
#
# Also any signature changes of these functions, as well as changes to the `GitHubPR`
# class, will likely require corresponding changes for the internal workflows.

import base64
import json
import os
import re
import time
import urllib.parse
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from re import Pattern
from typing import Any, cast, NamedTuple, Optional
from warnings import warn

import yaml
from github_utils import (
    gh_close_pr,
    gh_fetch_json_list,
    gh_fetch_merge_base,
    gh_fetch_url,
    gh_graphql,
    gh_post_commit_comment,
    gh_post_pr_comment,
    gh_update_pr_state,
    GitHubComment,
)
from gitutils import (
    are_ghstack_branches_in_sync,
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
    patterns_to_regex,
    retries_decorator,
)
from label_utils import (
    gh_add_labels,
    gh_remove_label,
    has_required_labels,
    LABEL_ERR_MSG,
)
from trymerge_explainer import get_revert_message, TryMergeExplainer


# labels
MERGE_IN_PROGRESS_LABEL = "merging"
MERGE_COMPLETE_LABEL = "merged"


class JobCheckState(NamedTuple):
    name: str
    url: str
    status: Optional[str]
    classification: Optional[str]
    job_id: Optional[int]
    title: Optional[str]
    summary: Optional[str]


JobNameToStateDict = dict[str, JobCheckState]


class WorkflowCheckState:
    def __init__(self, name: str, url: str, run_id: int, status: Optional[str]):
        self.name: str = name
        self.url: str = url
        self.run_id: int = run_id
        self.status: Optional[str] = status
        self.jobs: JobNameToStateDict = {}


GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
    bodyText
    createdAt
    authorAssociation
    editor {
      login
    }
    databaseId
    url
    state
  }
  pageInfo {
    startCursor
    hasPreviousPage
  }
}
"""

GH_CHECKSUITES_FRAGMENT = """
fragment PRCheckSuites on CheckSuiteConnection {
  edges {
    node {
      workflowRun {
        workflow {
          name
          databaseId
        }
        databaseId
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
          databaseId
          title
          summary
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
    }
    cursor
  }
  pageInfo {
    hasNextPage
  }
}
"""

GH_COMMIT_AUTHORS_FRAGMENT = """
fragment CommitAuthors on PullRequestCommitConnection {
  nodes {
    commit {
      authors(first: 2) {
        nodes {
          user {
            login
          }
          email
          name
        }
      }
      oid
    }
  }
  pageInfo {
    endCursor
    hasNextPage
  }
}
"""

GH_GET_PR_INFO_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + GH_CHECKSUITES_FRAGMENT
    + GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closed
      isCrossRepository
      author {
        login
      }
      title
      body
      headRefName
      headRepository {
        nameWithOwner
      }
      baseRefName
      baseRefOid
      baseRepository {
        nameWithOwner
        isPrivate
        defaultBranchRef {
          name
        }
      }
      mergeCommit {
        oid
      }
      commits_with_authors: commits(first: 100) {
        ...CommitAuthors
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              ...PRCheckSuites
            }
            status {
              contexts {
                context
                state
                targetUrl
              }
            }
            oid
          }
        }
      }
      changedFiles
      files(first: 100) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      reviews(last: 100) {
        ...PRReviews
      }
      comments(last: 5) {
        nodes {
          bodyText
          createdAt
          author {
            login
            url
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
      labels(first: 100) {
        edges {
          node {
            name
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_FILES_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      files(first: 100, after: $cursor) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_CHECKSUITES = (
    GH_CHECKSUITES_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              ...PRCheckSuites
            }
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 1, after: $cs_cursor) {
              nodes {
                checkRuns(first: 100, after: $cr_cursor) {
                  nodes {
                    name
                    conclusion
                    detailsUrl
                    databaseId
                    title
                    summary
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""

# This query needs read-org permission
GH_GET_TEAM_MEMBERS_QUERY = """
query($org: String!, $name: String!, $cursor: String) {
  organization(login: $org) {
    team(slug: $name) {
      members(first: 100, after: $cursor) {
        nodes {
          login
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_AUTHORS_QUERY = (
    GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
        ...CommitAuthors
      }
    }
  }
}
"""
)

GH_GET_PR_PREV_REVIEWS_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      reviews(last: 100, before: $cursor) {
        ...PRReviews
      }
    }
  }
}
"""
)

GH_GET_REPO_SUBMODULES = """
query ($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    submodules(first: 100) {
      nodes {
        path
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r"Stack.*:\r?\n(\* [^\r\n]+\r?\n)+", re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r"(Pull Request resolved|Pull-Request-resolved|Pull-Request): "
    r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)",
    re.MULTILINE,
)
RE_PR_CC_LINE = re.compile(r"^cc:? @\w+.*\r?\n?$", re.MULTILINE)
RE_DIFF_REV = re.compile(r"^Differential Revision:.+?(D[0-9]+)", re.MULTILINE)
CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")
MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"
REMOTE_MAIN_BRANCH = "origin/main"
DRCI_CHECKRUN_NAME = "Dr.CI"
INTERNAL_CHANGES_CHECKRUN_NAME = "Meta Internal-Only Changes Check"
HAS_NO_CONNECTED_DIFF_TITLE = (
    "There is no internal Diff connected, this can be merged now"
)
# This could be set to -1 to ignore all flaky and broken trunk failures. On the
# other hand, using a large value like 10 here might be useful in sev situation
IGNORABLE_FAILED_CHECKS_THESHOLD = 10


def iter_issue_timeline_until_comment(
    org: str, repo: str, issue_number: int, target_comment_id: int, max_pages: int = 200
) -> Any:
    """
    Yield timeline entries in order until (and including) the entry whose id == target_comment_id
    for a 'commented' event. Stops once the target comment is encountered.
    """
    page = 1

    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{org}/{repo}/issues/{issue_number}/timeline"
        )
        params = {"per_page": 100, "page": page}

        batch = gh_fetch_json_list(url, params)

        if not batch:
            return
        for ev in batch:
            # The target is the issue comment row with event == "commented" and id == issue_comment_id
            if ev.get("event") == "commented" and ev.get("id") == target_comment_id:
                yield ev  # nothing in the timeline after this matters, so stop early
                return
            yield ev
        if len(batch) < 100:
            return
        page += 1

    # If we got here without finding the comment, then we either hit a bug or some github PR
    # has a _really_ long timeline.
    # The max # of pages found on any pytorch/pytorch PR at the time of this change was 41
    raise RuntimeError(
        f"Could not find a merge commit in the first {max_pages} pages of the timeline at url {url}."
        f"This is most likely a bug, please report it to the @pytorch/pytorch-dev-infra team."
    )


def sha_from_committed_event(ev: dict[str, Any]) -> Optional[str]:
    """Extract SHA from committed event in timeline"""
    return ev.get("sha")


def sha_from_force_push_after(ev: dict[str, Any]) -> Optional[str]:
    """Extract SHA from force push event in timeline"""
    # The current GitHub API format
    commit_id = ev.get("commit_id")
    if commit_id:
        return str(commit_id)

    # Legacy format
    after = ev.get("after") or ev.get("after_commit") or {}
    if isinstance(after, dict):
        return after.get("sha") or after.get("oid")
    return ev.get("after_sha") or ev.get("head_sha")


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


@cache
def gh_get_team_members(org: str, name: str) -> list[str]:
    rc: list[str] = []
    team_members: dict[str, Any] = {
        "pageInfo": {"hasNextPage": "true", "endCursor": None}
    }
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(
            GH_GET_TEAM_MEMBERS_QUERY,
            org=org,
            name=name,
            cursor=team_members["pageInfo"]["endCursor"],
        )
        team = query["data"]["organization"]["team"]
        if team is None:
            warn(f"Requested non-existing team {org}/{name}")
            return []
        team_members = team["members"]
        rc += [member["login"] for member in team_members["nodes"]]
    return rc


def get_check_run_name_prefix(workflow_run: Any) -> str:
    if workflow_run is None:
        return ""
    else:
        return f"{workflow_run['workflow']['name']} / "


def is_passing_status(status: Optional[str]) -> bool:
    return status is not None and status.upper() in ["SUCCESS", "SKIPPED", "NEUTRAL"]


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[list[dict[str, dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any],
) -> JobNameToStateDict:
    # graphql seems to favor the most recent workflow run, so in theory we
    # shouldn't need to account for reruns, but do it just in case

    # workflow -> job -> job info
    workflows: dict[str, WorkflowCheckState] = {}

    # for the jobs that don't have a workflow
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState("", "", 0, None)

    def add_conclusions(edges: Any) -> None:
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]

            workflow_obj: WorkflowCheckState = no_workflow_obj

            if workflow_run is not None:
                # This is the usual workflow run ID we see on GitHub
                workflow_run_id = workflow_run["databaseId"]
                # While this is the metadata name and ID of the workflow itself
                workflow_name = workflow_run["workflow"]["name"]
                workflow_id = workflow_run["workflow"]["databaseId"]

                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in workflows:
                    continue

                # Only keep the latest workflow run for each workflow, heuristically,
                # it's the run with largest run ID
                if (
                    workflow_id not in workflows
                    or workflows[workflow_id].run_id < workflow_run_id
                ):
                    workflows[workflow_id] = WorkflowCheckState(
                        name=workflow_name,
                        status=workflow_conclusion,
                        url=workflow_run["url"],
                        run_id=workflow_run_id,
                    )
                workflow_obj = workflows[workflow_id]

            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    checkrun_name = f"{get_check_run_name_prefix(workflow_run)}{checkrun_node['name']}"
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(
                        existing_checkrun.status
                    ):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            checkrun_name,
                            checkrun_node["detailsUrl"],
                            checkrun_node["conclusion"],
                            classification=None,
                            job_id=checkrun_node["databaseId"],
                            title=checkrun_node["title"],
                            summary=checkrun_node["summary"],
                        )

                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None

    all_edges = checksuites["edges"].copy()
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        all_edges.extend(checksuites["edges"])

    add_conclusions(all_edges)

    # Flatten the dictionaries.  If there exists jobs in the workflow run, put
    # the jobs in but don't put the workflow in.  We care more about the jobs in
    # the workflow that ran than the container workflow.
    res: JobNameToStateDict = {}
    for workflow in workflows.values():
        if len(workflow.jobs) > 0:
            for job_name, job in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow.name] = JobCheckState(
                workflow.name,
                workflow.url,
                workflow.status,
                classification=None,
                job_id=None,
                title=None,
                summary=None,
            )
    for job_name, job in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-current", action="store_true")
    parser.add_argument("--check-mergeability", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("--reason", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def can_skip_internal_checks(pr: "GitHubPR", comment_id: Optional[int] = None) -> bool:
    if comment_id is None:
        return False
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return False
    return comment.author_login == "facebook-github-bot"


def _revlist_to_prs(
    repo: GitRepo,
    pr: "GitHubPR",
    rev_list: Iterable[str],
    should_skip: Optional[Callable[[int, "GitHubPR"], bool]] = None,
) -> list[tuple["GitHubPR", str]]:
    rc: list[tuple[GitHubPR, str]] = []
    for idx, rev in enumerate(rev_list):
        msg = repo.commit_message(rev)
        # findall doesn't return named captures, so we need to use finditer
        all_matches = list(RE_PULL_REQUEST_RESOLVED.finditer(msg))
        if len(all_matches) != 1:
            raise RuntimeError(
                f"Found an unexpected number of PRs mentioned in commit {rev}: "
                f"{len(all_matches)}.  This is probably because you are using an "
                "old version of ghstack.  Please update ghstack and resubmit "
                "your PRs"
            )

        m = all_matches[0]
        if pr.org != m.group("owner") or pr.project != m.group("repo"):
            raise RuntimeError(
                f"PR {m.group('number')} resolved to wrong owner/repo pair"
            )
        pr_num = int(m.group("number"))
        candidate = GitHubPR(pr.org, pr.project, pr_num) if pr_num != pr.pr_num else pr
        if should_skip is not None and should_skip(idx, candidate):
            continue
        rc.append((candidate, rev))
    return rc


def get_ghstack_prs(
    repo: GitRepo, pr: "GitHubPR", open_only: bool = True
) -> list[tuple["GitHubPR", str]]:
    """
    Get the PRs in the stack that are below this PR (inclusive).  Throws error if any of the open PRs are out of sync.
    @:param open_only: Only return open PRs
    """
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")

    def skip_func(idx: int, candidate: "GitHubPR") -> bool:
        if not open_only or not candidate.is_closed():
            return False
        print(
            f"Skipping {idx + 1} of {len(rev_list)} PR (#{candidate.pr_num}) as its already been merged"
        )
        return True

    assert pr.is_ghstack_pr()
    entire_stack = _revlist_to_prs(repo, pr, reversed(rev_list), skip_func)
    print(
        f"Found {len(entire_stack)} PRs in the stack for {pr.pr_num}: {[x[0].pr_num for x in entire_stack]}"
    )

    for stacked_pr, rev in entire_stack:
        if stacked_pr.is_closed():
            continue
        base_ref = stacked_pr.base_ref()
        if base_ref == pr.default_branch():
            base_ref = repo.get_merge_base(
                f"{repo.remote}/{base_ref}", f"{repo.remote}/{stacked_pr.head_ref()}"
            )
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref(), base_ref):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on "
                + f"branch {stacked_pr.get_ghstack_orig_ref()} that would be merged into {stacked_pr.default_branch()}.  "
                + "This usually happens because there is a non ghstack change in the PR.  "
                + f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: Optional[list[str]] = None
        self.labels: Optional[list[str]] = None
        self.conclusions: Optional[JobNameToStateDict] = None
        self.comments: Optional[list[GitHubComment]] = None
        self._authors: Optional[list[tuple[str, str]]] = None
        self._reviews: Optional[list[tuple[str, str]]] = None
        self.merge_base: Optional[str] = None
        self.submodules: Optional[list[str]] = None

    def is_closed(self) -> bool:
        return bool(self.info["closed"])

    def is_cross_repo(self) -> bool:
        return bool(self.info["isCrossRepository"])

    def base_ref(self) -> str:
        return cast(str, self.info["baseRefName"])

    def default_branch(self) -> str:
        return cast(str, self.info["baseRepository"]["defaultBranchRef"]["name"])

    def head_ref(self) -> str:
        return cast(str, self.info["headRefName"])

    def is_ghstack_pr(self) -> bool:
        return RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

    def get_ghstack_orig_ref(self) -> str:
        assert self.is_ghstack_pr()
        return re.sub(r"/head$", "/orig", self.head_ref())

    def is_base_repo_private(self) -> bool:
        return bool(self.info["baseRepository"]["isPrivate"])

    def get_changed_files_count(self) -> int:
        return int(self.info["changedFiles"])

    def last_commit(self) -> Any:
        return self.info["commits"]["nodes"][-1]["commit"]

    def last_commit_sha(self, default: Optional[str] = None) -> str:
        # for commits, the oid is the sha

        if default is None:
            return str(self.last_commit()["oid"])

        return str(self.last_commit().get("oid", default))

    def get_merge_base(self) -> str:
        if self.merge_base:
            return self.merge_base

        last_commit_sha = self.last_commit_sha()
        # NB: We could use self.base_ref() here for regular PR, however, that doesn't
        # work for ghstack where the base is the custom branch, i.e. gh/USER/ID/base,
        # so let's just use main instead
        self.merge_base = gh_fetch_merge_base(
            self.org, self.project, last_commit_sha, self.default_branch()
        )

        # Fallback to baseRefOid if the API call fails, i.e. rate limit. Note that baseRefOid
        # points to the base ref associated with the PR or, in other words, the head of main
        # when the PR is created or rebased. This is not necessarily the merge base commit,
        # but it could serve as a fallback in most cases and it's readily available as part
        # of the PR info
        if not self.merge_base:
            self.merge_base = cast(str, self.info["baseRefOid"])

        return self.merge_base

    def get_changed_files(self) -> list[str]:
        if self.changed_files is None:
            info = self.info
            unique_changed_files = set()
            # Do not try to fetch more than 10K files
            for _ in range(100):
                unique_changed_files.update([x["path"] for x in info["files"]["nodes"]])
                if not info["files"]["pageInfo"]["hasNextPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_NEXT_FILES_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["files"]["pageInfo"]["endCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
            self.changed_files = list(unique_changed_files)

        if len(self.changed_files) != self.get_changed_files_count():
            raise RuntimeError("Changed file count mismatch")
        return self.changed_files

    def get_submodules(self) -> list[str]:
        if self.submodules is None:
            rc = gh_graphql(GH_GET_REPO_SUBMODULES, name=self.project, owner=self.org)
            info = rc["data"]["repository"]["submodules"]
            self.submodules = [s["path"] for s in info["nodes"]]
        return self.submodules

    def get_changed_submodules(self) -> list[str]:
        submodules = self.get_submodules()
        return [f for f in self.get_changed_files() if f in submodules]

    def has_invalid_submodule_updates(self) -> bool:
        """Submodule updates in PR are invalid if submodule keyword
        is not mentioned in neither the title nor body/description
        nor in any of the labels.
        """
        return (
            len(self.get_changed_submodules()) > 0
            and "submodule" not in self.get_title().lower()
            and "submodule" not in self.get_body().lower()
            and all("submodule" not in label for label in self.get_labels())
        )

    def _get_reviews(self) -> list[tuple[str, str]]:
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info["reviews"]["nodes"]
                self._reviews = [
                    (node["author"]["login"], node["state"]) for node in nodes
                ] + self._reviews
                if not info["reviews"]["pageInfo"]["hasPreviousPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_PREV_REVIEWS_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["reviews"]["pageInfo"]["startCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
        reviews = {
            author: state for author, state in self._reviews if state != "COMMENTED"
        }
        return list(reviews.items())

    def get_approved_by(self) -> list[str]:
        return [login for (login, state) in self._get_reviews() if state == "APPROVED"]

    def get_commit_count(self) -> int:
        return int(self.info["commits_with_authors"]["totalCount"])

    def get_commit_sha_at_comment(self, comment_id: int) -> Optional[str]:
        """
        Get the PR head commit SHA that was present when a specific comment was posted.
        This ensures we only merge the state of the PR at the time the merge command was issued,
        not any subsequent commits that may have been pushed after.

        Returns None if no head-changing events found before the comment or if the comment was not found.
        """
        head = None

        try:
            for event in iter_issue_timeline_until_comment(
                self.org, self.project, self.pr_num, comment_id
            ):
                etype = event.get("event")
                if etype == "committed":
                    sha = sha_from_committed_event(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found commit event for SHA {sha}")
                elif etype == "head_ref_force_pushed":
                    sha = sha_from_force_push_after(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found force push event for SHA {sha}")
                elif etype == "commented":
                    if event.get("id") == comment_id:
                        print(f"Timeline: Found final comment with sha {sha}")
                        return head
        except Exception as e:
            print(
                f"Warning: Failed to reconstruct timeline for comment {comment_id}: {e}"
            )
            return None

        print(f"Did not find comment with id {comment_id} in the PR timeline")
        return None

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def _fetch_authors(self) -> list[tuple[str, str]]:
        if self._authors is not None:
            return self._authors
        authors: list[tuple[str, str]] = []

        def add_authors(info: dict[str, Any]) -> None:
            for node in info["commits_with_authors"]["nodes"]:
                for author_node in node["commit"]["authors"]["nodes"]:
                    user_node = author_node["user"]
                    author = f"{author_node['name']} <{author_node['email']}>"
                    if user_node is None:
                        # If author is not github user, user node will be null
                        authors.append(("", author))
                    else:
                        authors.append((cast(str, user_node["login"]), author))

        info = self.info
        for _ in range(100):
            add_authors(info)
            if not info["commits_with_authors"]["pageInfo"]["hasNextPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_NEXT_AUTHORS_QUERY,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["commits_with_authors"]["pageInfo"]["endCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
        self._authors = authors
        return authors

    def get_committer_login(self, num: int = 0) -> str:
        return self._fetch_authors()[num][0]

    def get_committer_author(self, num: int = 0) -> str:
        return self._fetch_authors()[num][1]

    def get_labels(self) -> list[str]:
        if self.labels is not None:
            return self.labels
        labels = (
            [node["node"]["name"] for node in self.info["labels"]["edges"]]
            if "labels" in self.info
            else []
        )
        self.labels = labels
        return self.labels

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        """Returns dict of checkrun -> [conclusion, url]"""
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.last_commit()

        def get_pr_next_check_runs(
            edges: list[dict[str, dict[str, Any]]], edge_idx: int, checkruns: Any
        ) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECK_RUNS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                cr_cursor=checkruns["pageInfo"]["endCursor"],
            )
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][
                -1
            ]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECKSUITES,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=checksuites["edges"][-1]["cursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            return last_commit["checkSuites"]

        checksuites = orig_last_commit["checkSuites"]

        self.conclusions = add_workflow_conclusions(
            checksuites, get_pr_next_check_runs, get_pr_next_checksuites
        )

        # Append old style statuses(like ones populated by CircleCI or EasyCLA) to conclusions
        if orig_last_commit["status"] and orig_last_commit["status"]["contexts"]:
            for status in orig_last_commit["status"]["contexts"]:
                name = status["context"]
                self.conclusions[name] = JobCheckState(
                    name,
                    status["targetUrl"],
                    status["state"],
                    classification=None,
                    job_id=None,
                    title=None,
                    summary=None,
                )

        # Making an exception for Apply lint auggestions/autoformat because the
        # bot adds a merged label -> triggers workflow -> sometimes needs
        # approval -> is read as failure, which results in a blocked merge, but
        # this workflow doesn't provide mergability info
        self.conclusions.pop("Apply lint suggestions", None)

        return self.conclusions

    def get_authors(self) -> dict[str, str]:
        rc = {}
        for idx in range(len(self._fetch_authors())):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)

        return rc

    def get_author(self) -> str:
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        creator = self.get_pr_creator_login()
        # If PR creator is not among authors
        # Assume it was authored by first commit author
        if creator not in authors:
            return self.get_committer_author(0)
        return authors[creator]

    def get_title(self) -> str:
        return cast(str, self.info["title"])

    def get_body(self) -> str:
        return cast(str, self.info["body"])

    def get_merge_commit(self) -> Optional[str]:
        mc = self.info["mergeCommit"]
        return mc["oid"] if mc is not None else None

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        editor = node["editor"]
        return GitHubComment(
            body_text=node["bodyText"],
            created_at=node.get("createdAt", ""),
            author_login=node["author"]["login"],
            author_url=node["author"].get("url", None),
            author_association=node["authorAssociation"],
            editor_login=editor["login"] if editor else None,
            database_id=node["databaseId"],
            url=node["url"],
        )

    def get_comments(self) -> list[GitHubComment]:
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info["comments"]
        # Do not try to fetch more than 10K comments
        for _ in range(100):
            self.comments = [
                self._comment_from_node(node) for node in info["nodes"]
            ] + self.comments
            if not info["pageInfo"]["hasPreviousPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_PREV_COMMENTS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["pageInfo"]["startCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]["comments"]
        return self.comments

    def get_last_comment(self) -> GitHubComment:
        return self._comment_from_node(self.info["comments"]["nodes"][-1])

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        if self.comments is None:
            # Fastpath - try searching in partial prefetched comments
            for node in self.info["comments"]["nodes"]:
                comment = self._comment_from_node(node)
                if comment.database_id == database_id:
                    return comment

        for comment in self.get_comments():
            if comment.database_id == database_id:
                return comment

        # The comment could have actually been a review left on the PR (the message written alongside the review).
        # (This is generally done to trigger the merge right when a comment is left)
        # Check those review comments to see if one of those was the comment in question.
        for node in self.info["reviews"]["nodes"]:
            # These review comments contain all the fields regular comments need
            comment = self._comment_from_node(node)
            if comment.database_id == database_id:
                return comment

        raise RuntimeError(f"Comment with id {database_id} not found")

    def get_diff_revision(self) -> Optional[str]:
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != "SUCCESS"

    def has_no_connected_diff(self) -> bool:
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].title == HAS_NO_CONNECTED_DIFF_TITLE

    def merge_ghstack_into(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool,
        comment_id: Optional[int] = None,
        skip_all_rule_checks: bool = False,
    ) -> list["GitHubPR"]:
        assert self.is_ghstack_pr()
        ghstack_prs = get_ghstack_prs(
            repo, self, open_only=False
        )  # raises error if out of sync
        pr_dependencies = []
        for pr, rev in ghstack_prs:
            if pr.is_closed():
                pr_dependencies.append(pr)
                continue

            commit_msg = pr.gen_commit_message(
                filter_ghstack=True, ghstack_deps=pr_dependencies
            )
            if pr.pr_num != self.pr_num and not skip_all_rule_checks:
                # Raises exception if matching rule is not found
                find_matching_merge_rule(
                    pr,
                    repo,
                    skip_mandatory_checks=skip_mandatory_checks,
                    skip_internal_checks=can_skip_internal_checks(self, comment_id),
                )
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
            pr_dependencies.append(pr)
        return [x for x, _ in ghstack_prs if not x.is_closed()]

    def gen_commit_message(
        self,
        filter_ghstack: bool = False,
        ghstack_deps: Optional[list["GitHubPR"]] = None,
    ) -> str:
        """Fetches title and body from PR description
        adds reviewed by, pull request resolved and optionally
        filters out ghstack info"""
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ", ".join(
            prefix_with_github_url(login) for login in self.get_approved_by()
        )
        # Remove "cc: " line from the message body
        msg_body = re.sub(RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body

        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        if ghstack_deps:
            msg += f"ghstack dependencies: {', '.join([f'#{pr.pr_num}' for pr in ghstack_deps])}\n"

        # Mention PR co-authors, which should be at the end of the message
        # And separated from the body by two newlines
        first_coauthor = True
        for author_login, author_name in self.get_authors().items():
            if author_login != self.get_pr_creator_login():
                if first_coauthor:
                    msg, first_coauthor = (msg + "\n", False)
                msg += f"\nCo-authored-by: {author_name}"

        return msg

    def add_numbered_label(self, label_base: str, dry_run: bool) -> None:
        labels = self.get_labels() if self.labels is not None else []
        full_label = label_base
        count = 0
        for label in labels:
            if label_base in label:
                count += 1
                full_label = f"{label_base}X{count}"
        self.add_label(full_label, dry_run)

    def add_label(self, label: str, dry_run: bool) -> None:
        gh_add_labels(self.org, self.project, self.pr_num, [label], dry_run)

    def merge_into(
        self,
        repo: GitRepo,
        *,
        skip_mandatory_checks: bool = False,
        dry_run: bool = False,
        comment_id: int,
        ignore_current_checks: Optional[list[str]] = None,
    ) -> None:
        # Raises exception if matching rule is not found
        (
            merge_rule,
            pending_checks,
            failed_checks,
            ignorable_checks,
        ) = find_matching_merge_rule(
            self,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=can_skip_internal_checks(self, comment_id),
            ignore_current_checks=ignore_current_checks,
        )
        additional_merged_prs = self.merge_changes_locally(
            repo, skip_mandatory_checks, comment_id
        )

        repo.push(self.default_branch(), dry_run)
        if not dry_run:
            self.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)
            for pr in additional_merged_prs:
                pr.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)

        # When the merge process reaches this part, we can assume that the commit
        # has been successfully pushed to trunk
        merge_commit_sha = repo.rev_parse(name=self.default_branch())

        if comment_id and self.pr_num:
            # Finally, upload the record to s3. The list of pending and failed
            # checks are at the time of the merge
            save_merge_record(
                comment_id=comment_id,
                pr_num=self.pr_num,
                owner=self.org,
                project=self.project,
                author=self.get_author(),
                pending_checks=pending_checks,
                failed_checks=failed_checks,
                ignore_current_checks=ignorable_checks.get("IGNORE_CURRENT_CHECK", []),
                broken_trunk_checks=ignorable_checks.get("BROKEN_TRUNK", []),
                flaky_checks=ignorable_checks.get("FLAKY", []),
                unstable_checks=ignorable_checks.get("UNSTABLE", []),
                last_commit_sha=self.last_commit_sha(default=""),
                merge_base_sha=self.get_merge_base(),
                merge_commit_sha=merge_commit_sha,
                is_failed=False,
                skip_mandatory_checks=skip_mandatory_checks,
                ignore_current=bool(ignore_current_checks),
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to s3")

        # Usually Github will see that the commit has "resolves <pr_num>" in the
        # commit message and close the PR, but sometimes it doesn't, leading to
        # confusion.  When it doesn't, we close it manually.
        time.sleep(60)  # Give Github some time to close the PR
        manually_close_merged_pr(
            pr=self,
            additional_merged_prs=additional_merged_prs,
            merge_commit_sha=merge_commit_sha,
            dry_run=dry_run,
        )

    def merge_changes_locally(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool = False,
        comment_id: Optional[int] = None,
        branch: Optional[str] = None,
        skip_all_rule_checks: bool = False,
    ) -> list["GitHubPR"]:
        """
        :param skip_all_rule_checks: If true, skips all rule checks on ghstack PRs, useful for dry-running merge locally
        """
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)

        # It's okay to skip the commit SHA check for ghstack PRs since
        # authoring requires write access to the repo.
        if self.is_ghstack_pr():
            return self.merge_ghstack_into(
                repo,
                skip_mandatory_checks,
                comment_id=comment_id,
                skip_all_rule_checks=skip_all_rule_checks,
            )

        msg = self.gen_commit_message()
        pr_branch_name = f"__pull-request-{self.pr_num}__init__"

        # Determine which commit SHA to merge
        commit_to_merge = None
        if not comment_id:
            raise ValueError("Must provide --comment-id when merging regular PRs")

        # Get the commit SHA that was present when the comment was made
        commit_to_merge = self.get_commit_sha_at_comment(comment_id)
        if not commit_to_merge:
            raise RuntimeError(
                f"Could not find commit that was pushed before comment {comment_id}"
            )

        # Validate that this commit is the latest commit on the PR
        latest_commit = self.last_commit_sha()
        if commit_to_merge != latest_commit:
            raise RuntimeError(
                f"Commit {commit_to_merge} was HEAD when comment {comment_id} was posted "
                f"but now the latest commit on the PR is {latest_commit}. "
                f"Please re-issue the merge command to merge the latest commit."
            )

        print(f"Merging commit {commit_to_merge} locally")

        repo.fetch(commit_to_merge, pr_branch_name)
        repo._run_git("merge", "--squash", pr_branch_name)
        repo._run_git("commit", f'--author="{self.get_author()}"', "-m", msg)

        # Did the PR change since we started the merge?
        pulled_sha = repo.show_ref(pr_branch_name)
        latest_pr_status = GitHubPR(self.org, self.project, self.pr_num)
        if (
            pulled_sha != latest_pr_status.last_commit_sha()
            or pulled_sha != commit_to_merge
        ):
            raise RuntimeError(
                "PR has been updated since CI checks last passed. Please rerun the merge command."
            )
        return []


class MergeRuleFailedError(RuntimeError):
    def __init__(self, message: str, rule: Optional["MergeRule"] = None) -> None:
        super().__init__(message)
        self.rule = rule


class MandatoryChecksMissingError(MergeRuleFailedError):
    pass


class PostCommentError(Exception):
    pass


@dataclass
class MergeRule:
    name: str
    patterns: list[str]
    approved_by: list[str]
    mandatory_checks_name: Optional[list[str]]
    ignore_flaky_failures: bool = True


def gen_new_issue_link(
    org: str, project: str, labels: list[str], template: str = "bug-report.yml"
) -> str:
    labels_str = ",".join(labels)
    return (
        f"https://github.com/{org}/{project}/issues/new?"
        f"labels={urllib.parse.quote(labels_str)}&"
        f"template={urllib.parse.quote(template)}"
    )


def read_merge_rules(
    repo: Optional[GitRepo], org: str, project: str
) -> list[MergeRule]:
    """Returns the list of all merge rules for the repo or project.

    NB: this function is used in Meta-internal workflows, see the comment
    at the top of this file for details.
    """
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = gh_fetch_url(
            f"https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}",
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )
        content = base64.b64decode(json_data["content"])
        return [MergeRule(**x) for x in yaml.safe_load(content)]
    else:
        rules_path = Path(repo.repo_dir) / repo_relative_rules_path
        if not rules_path.exists():
            print(f"{rules_path} does not exist, returning empty rules")
            return []
        with open(rules_path) as fp:
            rc = yaml.safe_load(fp)
        return [MergeRule(**x) for x in rc]


def find_matching_merge_rule(
    pr: GitHubPR,
    repo: Optional[GitRepo] = None,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: Optional[list[str]] = None,
) -> tuple[
    MergeRule,
    list[tuple[str, Optional[str], Optional[int]]],
    list[tuple[str, Optional[str], Optional[int]]],
    dict[str, list[Any]],
]:
    """
    Returns merge rule matching to this pr together with the list of associated pending
    and failing jobs OR raises an exception.

    NB: this function is used in Meta-internal workflows, see the comment at the top of
    this file for details.
    """
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())

    issue_link = 
```



## High-Level Overview


This Python file contains 8 class(es) and 95 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `JobCheckState`, `WorkflowCheckState`, `GitHubPR`, `MergeRuleFailedError`, `MandatoryChecksMissingError`, `PostCommentError`, `MergeRule`

**Functions defined**: `__init__`, `iter_issue_timeline_until_comment`, `sha_from_committed_event`, `sha_from_force_push_after`, `gh_get_pr_info`, `gh_get_team_members`, `get_check_run_name_prefix`, `is_passing_status`, `add_workflow_conclusions`, `add_conclusions`, `parse_args`, `can_skip_internal_checks`, `_revlist_to_prs`, `get_ghstack_prs`, `skip_func`, `__init__`, `is_closed`, `is_cross_repo`, `base_ref`, `default_branch`

**Key imports**: base64, json, os, re, time, urllib.parse, defaultdict, Callable, Iterable, dataclass, cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `base64`
- `json`
- `os`
- `re`
- `time`
- `urllib.parse`
- `collections`: defaultdict
- `collections.abc`: Callable, Iterable
- `dataclasses`: dataclass
- `functools`: cache
- `pathlib`: Path
- `typing`: Any, cast, NamedTuple, Optional
- `warnings`: warn
- `yaml`
- `trymerge_explainer`: get_revert_message, TryMergeExplainer
- `argparse`: ArgumentParser
- `traceback`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.github/scripts`):

- [`convert_lintrunner_annotations_to_github.py_docs.md`](./convert_lintrunner_annotations_to_github.py_docs.md)
- [`gitutils.py_docs.md`](./gitutils.py_docs.md)
- [`collect_ciflow_labels.py_docs.md`](./collect_ciflow_labels.py_docs.md)
- [`generate_docker_release_matrix.py_docs.md`](./generate_docker_release_matrix.py_docs.md)
- [`github_utils.py_docs.md`](./github_utils.py_docs.md)
- [`filter_test_configs.py_docs.md`](./filter_test_configs.py_docs.md)
- [`test_runner_determinator.py_docs.md`](./test_runner_determinator.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `trymerge.py_docs.md`
- **Keyword Index**: `trymerge.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
