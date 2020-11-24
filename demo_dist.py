
import base64
import json
import os
import sys
import subprocess
import graphscope
import time

from graphscope.framework.loader import Loader

def load_ldbc10k(sess, prefix):
    vertices = {
        "person": (
            Loader(
                os.path.join(prefix, "person_degree.csv"), header_row=True, delimiter="|"
            ),
            [],
            0,
        ),
    }

    edges = {
        "knows": [
            (
                Loader(
                    os.path.join(prefix, "person_knows_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["creationDate"],
                ("Person.id", "person"),
                ("Person.id.1", "person"),
            )
        ],
    }
    g = sess.load_from(edges, vertices)
    return g

def ldbc_50(sess, prefix):
    """Load ldbc 50 dataset as a ArrowProperty Graph.

    Args:
        sess (graphscope.Session): Load graph within the session.
        prefix (str): Data directory.

    Returns:
        graphscope.Graph: A Graph object which graph type is ArrowProperty
    """
    vertices = {
        "comment": (
            Loader(
                os.path.join(prefix, "comment_0_0.csv"), header_row=True, delimiter="|"
            ),
            ["creationDate", "locationIP", "browserUsed", "content", "length"],
            "id",
        ),
        "organisation": (
            Loader(
                os.path.join(prefix, "organisation_0_0.csv"),
                header_row=True,
                delimiter="|",
            ),
            ["type", "name", "url"],
            "id",
        ),
        "tagclass": (
            Loader(
                os.path.join(prefix, "tagclass_0_0.csv"), header_row=True, delimiter="|"
            ),
            ["name", "url"],
            "id",
        ),
        "person": (
            Loader(
                os.path.join(prefix, "person_0_0.csv"), header_row=True, delimiter="|"
            ),
            [
                "firstName",
                "lastName",
                "gender",
                "birthday",
                "creationDate",
                "locationIP",
                "browserUsed",
            ],
            "id",
        ),
        "forum": (
            Loader(
                os.path.join(prefix, "forum_0_0.csv"), header_row=True, delimiter="|"
            ),
            ["title", "creationDate"],
            "id",
        ),
        "place": (
            Loader(
                os.path.join(prefix, "place_0_0.csv"), header_row=True, delimiter="|"
            ),
            ["name", "url", "type"],
            "id",
        ),
        "post": (
            Loader(
                os.path.join(prefix, "post_0_0.csv"), header_row=True, delimiter="|"
            ),
            [
                "imageFile",
                "creationDate",
                "locationIP",
                "browserUsed",
                # "language",
                # "content",
                "length",
            ],
            "id",
        ),
        "tag": (
            Loader(os.path.join(prefix, "tag_0_0.csv"), header_row=True, delimiter="|"),
            ["name", "url"],
            "id",
        ),
    }
    edges = {
        "replyOf": [
            (
                Loader(
                    os.path.join(prefix, "comment_replyOf_comment_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Comment.id", "comment"),
                ("Comment.id.1", "comment"),
            ),
            (
                Loader(
                    os.path.join(prefix, "comment_replyOf_post_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Comment.id", "comment"),
                ("Post.id", "post"),
            ),
        ],
        "isPartOf": [
            (
                Loader(
                    os.path.join(prefix, "place_isPartOf_place_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Place.id", "place"),
                ("Place.id.1", "place"),
            )
        ],
        "isSubclassOf": [
            (
                Loader(
                    os.path.join(prefix, "tagclass_isSubclassOf_tagclass_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("TagClass.id", "tagclass"),
                ("TagClass.id.1", "tagclass"),
            )
        ],
        "hasTag": [
            (
                Loader(
                    os.path.join(prefix, "forum_hasTag_tag_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Forum.id", "forum"),
                ("Tag.id", "tag"),
            ),
            (
                Loader(
                    os.path.join(prefix, "comment_hasTag_tag_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Comment.id", "comment"),
                ("Tag.id", "tag"),
            ),
            (
                Loader(
                    os.path.join(prefix, "post_hasTag_tag_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Post.id", "post"),
                ("Tag.id", "tag"),
            ),
        ],
        "knows": [
            (
                Loader(
                    os.path.join(prefix, "person_knows_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "creationDate"],
                ("Person.id", "person"),
                ("Person.id.1", "person"),
            )
        ],
        "hasModerator": [
            (
                Loader(
                    os.path.join(prefix, "forum_hasModerator_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Forum.id", "forum"),
                ("Person.id", "person"),
            )
        ],
        "hasInterest": [
            (
                Loader(
                    os.path.join(prefix, "person_hasInterest_tag_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Person.id", "person"),
                ("Tag.id", "tag"),
            )
        ],
        "isLocatedIn": [
            (
                Loader(
                    os.path.join(prefix, "post_isLocatedIn_place_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Post.id", "post"),
                ("Place.id", "place"),
            ),
            (
                Loader(
                    os.path.join(prefix, "comment_isLocatedIn_place_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Comment.id", "comment"),
                ("Place.id", "place"),
            ),
            (
                Loader(
                    os.path.join(prefix, "organisation_isLocatedIn_place_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Organisation.id", "organisation"),
                ("Place.id", "place"),
            ),
            (
                Loader(
                    os.path.join(prefix, "person_isLocatedIn_place_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Person.id", "person"),
                ("Place.id", "place"),
            ),
        ],
        "hasType": [
            (
                Loader(
                    os.path.join(prefix, "tag_hasType_tagclass_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Tag.id", "tag"),
                ("TagClass.id", "tagclass"),
            )
        ],
        "hasCreator": [
            (
                Loader(
                    os.path.join(prefix, "post_hasCreator_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Post.id", "post"),
                ("Person.id", "person"),
            ),
            (
                Loader(
                    os.path.join(prefix, "comment_hasCreator_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Comment.id", "comment"),
                ("Person.id", "person"),
            ),
        ],
        "containerOf": [
            (
                Loader(
                    os.path.join(prefix, "forum_containerOf_post_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                [
                    "eid_generated",
                ],
                ("Forum.id", "forum"),
                ("Post.id", "post"),
            )
        ],
        "hasMember": [
            (
                Loader(
                    os.path.join(prefix, "forum_hasMember_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "joinDate"],
                ("Forum.id", "forum"),
                ("Person.id", "person"),
            )
        ],
        "workAt": [
            (
                Loader(
                    os.path.join(prefix, "person_workAt_organisation_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "workFrom"],
                ("Person.id", "person"),
                ("Organisation.id", "organisation"),
            )
        ],
        "likes": [
            (
                Loader(
                    os.path.join(prefix, "person_likes_comment_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "creationDate"],
                ("Person.id", "person"),
                ("Comment.id", "comment"),
            ),
            (
                Loader(
                    os.path.join(prefix, "person_likes_post_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "creationDate"],
                ("Person.id", "person"),
                ("Post.id", "post"),
            ),
        ],
        "studyAt": [
            (
                Loader(
                    os.path.join(prefix, "person_studyAt_organisation_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "classYear"],
                ("Person.id", "person"),
                ("Organisation.id", "organisation"),
            )
        ],
    }
    return sess.load_from(edges, vertices)

# load ldbc-10k-person dataset
def load_ldbc50_subgraph(sess, prefix):
    """Load a subgraph of ldbc50, which only contains people-typed node
       and knows-type edge.

    Args:
        sess (graphscope.Session): Load graph within the session.
        prefix (str): Data directory.

    Returns:
        graphscope.Graph: A Graph object which graph type is ArrowProperty
    """
    vertices = {
        "person": (
            Loader(
                os.path.join(prefix, "person_0_0.csv"), header_row=True, delimiter="|"
            ),
            [
                "firstName",
                "lastName",
                "gender",
                "birthday",
                "creationDate",
                "locationIP",
                "browserUsed",
            ],
            "id",
        ),
    }

    edges = {
        "knows": [
            (
                Loader(
                    os.path.join(prefix, "person_knows_person_0_0.csv"),
                    header_row=True,
                    delimiter="|",
                ),
                ["eid_generated", "creationDate"],
                ("Person.id", "person"),
                ("Person.id.1", "person"),
            ),
        ],
    }
    g = sess.load_from(edges, vertices)
    return g

if __name__ == '__main__':

    # init a session with test namepace
    sess = graphscope.session(num_workers=1, enable_k8s=True,
                          k8s_gs_image="registry.cn-hongkong.aliyuncs.com/graphscope/graphscope:gl-demo-2",
                          k8s_vineyard_image="registry.cn-hongkong.aliyuncs.com/graphscope/graphscope:gl-demo-2")

    # get gl handle
    # prefix = "/root/gsa/gstest/ldbc_50"
    prefix = "/root/gsa/learning_engine/graph-learn/examples/data/ldbc_10k_people"
    graph = load_ldbc10k(sess, prefix)
    print(graph.schema)
    # graph = load_ldbc50_subgraph(sess, prefix)
    # graph = ldbc_50(sess, prefix)
    handle = sess.get_gl_handle(graph, 'person', 'knows', 2)

    # debug msg
    s = base64.b64decode(handle).decode('utf-8')
    obj = json.loads(s)
    print(obj)
    print('handle', handle)

    # run link_prediction
    servers = sess.info["engine_hosts"].split(",")
    namespace = sess._config_params["k8s_namespace"]
    print('namespce', namespace)
    time.sleep(100000)

    
    procs = []
    for task_index, pop in enumerate(servers):
        print("run task_index", task_index, pop)
        p = subprocess.Popen(
            [
                "kubectl",
                "exec",
                "--namespace",
                namespace,
                "--container",
                "engine",
                pop,
                "--",
                "python3",
                "/root/gsa/learning_engine/graph-learn/examples/tf/graphsage/link_prediction.py",
                handle,
                str(task_index),
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=0
        )
        procs.append(p)


    return_code = procs[0].wait()
    print('return code', return_code)
    sess.close()

