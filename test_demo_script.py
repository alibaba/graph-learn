import base64
import json
import os
import sys
import subprocess
import graphscope
from grpahscope.framework.loader import Loader


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


if __name__ == '__main__':
    sess = graphscope.session(num_workers=1, enable_k8s=True,
                              k8s_gs_image="registry.cn-hongkong.aliyuncs.com/graphscope/graphscope:gl-demo-3",
                              k8s_vineyard_image="registry.cn-hongkong.aliyuncs.com/graphscope/graphscope:gl-demo-3")

    prefix = "/root/gsa/gstest/ldbc_50"
    graph = ldbc_50(sess, prefix)

    # Analytical engine
    # project the projected graph to simple graph.
    # FIXME: Use sub_graph to project
    # simple_g = sub_graph.project_to_simple(v_label="person", e_label="knows")
    simple_g = graph.project_to_simple(v_label="person", e_label="knows")

    pr_result = graphscope.pagerank(simple_g, delta=0.8)
    tc_result = graphscope.triangles(simple_g)

    # add the PageRank and triangle-counting results as new columns to the property graph
    # FIXME: Add column to sub_graph
    graph.add_column(pr_result, {"Ranking": "r"})
    graph.add_column(tc_result, {"TC": "r"})

    # GNN engine
    # TODO(zhurong): run the link-prediction over the sub_graph
    # FIXME: get sub_graph handle
    gl_handle = sess.get_gl_handle(graph, client_number=4)
    namespace = sess._config_params['k8s_namespace']
    print('namespace', namespace)
    pop_name = sess.info["engine_hosts"]
    print('pop', pop_name)
    cmd = ["kubectl", "exec", '--namespace' 'gl-demo-test', '--container', 'engine', \
            pop_name, '--', 'python3', '/root/gsa/learning_engine/graph-learn/examples/tf/graphsage/link_prediction.py', gl_handle]

    subprocess.check_output(cmd, stdout=sys.stdout, stderr=sys.stderr, bufsize=0)
    # TODO: maybe need another python shown as below

    # TODO(zhurong): single machine training.
    # g1 = sess.gl(g)
    # g1 is a gl.Graph type
    sess.close()