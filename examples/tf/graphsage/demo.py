import os
import graphscope
import base64
import json
import sys
import time

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
            os.path.join(prefix, "comment_0_0.csv") + "#header_row=true&delimiter=|",
            ["creationDate", "locationIP", "browserUsed", "content", "length"],
            "id",
        ),
        "organisation": (
            os.path.join(prefix, "organisation_0_0.csv") + "#header_row=true&delimiter=|",
            ["type", "name", "url"],
            "id",
        ),
        "tagclass": (
            os.path.join(prefix, "tagclass_0_0.csv") + "#header_row=true&delimiter=|",
            ["name", "url"],
            "id",
        ),
        "train": (
            os.path.join(prefix, "person_0_0.csv") + "#header_row=true&delimiter=|",
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
            os.path.join(prefix, "forum_0_0.csv") + "#header_row=true&delimiter=|",
            ["title", "creationDate"],
            "id",
        ),
        "place": (
            os.path.join(prefix, "place_0_0.csv") + "#header_row=true&delimiter=|",
            ["name", "url", "type"],
            "id",
        ),
        "post": (
            os.path.join(prefix, "post_0_0.csv") + "#header_row=true&delimiter=|",
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
            os.path.join(prefix, "tag_0_0.csv") + "#header_row=true&delimiter=|",
            ["name", "url"],
            "id",
        ),
    }
    edges = {
        "replyOf": [
            (
                os.path.join(prefix, "comment_replyOf_comment_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Comment.id", "comment"),
                ("Comment.id.1", "comment"),
            ),
            (
                os.path.join(prefix, "comment_replyOf_post_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Comment.id", "comment"),
                ("Post.id", "post"),
            ),
        ],
        "isPartOf": [
            (
                os.path.join(prefix, "place_isPartOf_place_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Place.id", "place"),
                ("Place.id.1", "place"),
            )
        ],
        "isSubclassOf": [
            (
                os.path.join(prefix, "tagclass_isSubclassOf_tagclass_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("TagClass.id", "tagclass"),
                ("TagClass.id.1", "tagclass"),
            )
        ],
        "hasTag": [
            (
                os.path.join(prefix, "forum_hasTag_tag_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Forum.id", "forum"),
                ("Tag.id", "tag"),
            ),
            (
                os.path.join(prefix, "comment_hasTag_tag_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Comment.id", "comment"),
                ("Tag.id", "tag"),
            ),
            (
                os.path.join(prefix, "post_hasTag_tag_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Post.id", "post"),
                ("Tag.id", "tag"),
            ),
        ],
        "knows": [
            (
                os.path.join(prefix, "person_knows_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "creationDate"],
                ("Person.id", "train"),
                ("Person.id.1", "train"),
            )
        ],
        "hasModerator": [
            (
                os.path.join(prefix, "forum_hasModerator_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Forum.id", "forum"),
                ("Person.id", "train"),
            )
        ],
        "hasInterest": [
            (
                os.path.join(prefix, "person_hasInterest_tag_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Person.id", "train"),
                ("Tag.id", "tag"),
            )
        ],
        "isLocatedIn": [
            (
                os.path.join(prefix, "post_isLocatedIn_place_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Post.id", "post"),
                ("Place.id", "place"),
            ),
            (
                os.path.join(prefix, "comment_isLocatedIn_place_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Comment.id", "comment"),
                ("Place.id", "place"),
            ),
            (
                os.path.join(prefix, "organisation_isLocatedIn_place_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Organisation.id", "organisation"),
                ("Place.id", "place"),
            ),
            (
                os.path.join(prefix, "person_isLocatedIn_place_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Person.id", "train"),
                ("Place.id", "place"),
            ),
        ],
        "hasType": [
            (
                os.path.join(prefix, "tag_hasType_tagclass_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Tag.id", "tag"),
                ("TagClass.id", "tagclass"),
            )
        ],
        "hasCreator": [
            (
                os.path.join(prefix, "post_hasCreator_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Post.id", "post"),
                ("Person.id", "train"),
            ),
            (
                os.path.join(prefix, "comment_hasCreator_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Comment.id", "comment"),
                ("Person.id", "train"),
            ),
        ],
        "containerOf": [
            (
                os.path.join(prefix, "forum_containerOf_post_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", ],
                ("Forum.id", "forum"),
                ("Post.id", "post"),
            )
        ],
        "hasMember": [
            (
                os.path.join(prefix, "forum_hasMember_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "joinDate"],
                ("Forum.id", "forum"),
                ("Person.id", "train"),
            )
        ],
        "workAt": [
            (
                os.path.join(prefix, "person_workAt_organisation_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "workFrom"],
                ("Person.id", "train"),
                ("Organisation.id", "organisation"),
            )
        ],
        "likes": [
            (
                os.path.join(prefix, "person_likes_comment_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "creationDate"],
                ("Person.id", "train"),
                ("Comment.id", "comment"),
            ),
            (
                os.path.join(prefix, "person_likes_post_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "creationDate"],
                ("Person.id", "train"),
                ("Post.id", "post"),
            ),
        ],
        "studyAt": [
            (
                os.path.join(prefix, "person_studyAt_organisation_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "classYear"],
                ("Person.id", "train"),
                ("Organisation.id", "organisation"),
            )
        ],
    }
    return sess.load_from(edges, vertices)


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
        "item": (
            os.path.join(prefix, "person_0_0.csv") + "#header_row=true&delimiter=|",
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
        "train": (
            os.path.join(prefix, "person_0_0.csv") + "#header_row=true&delimiter=|",
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
        "relation": [
            (
                os.path.join(prefix, "person_knows_person_0_0.csv") + "#header_row=true&delimiter=|",
                ["eid_generated", "creationDate"],
                ("Person.id", "item"),
                ("Person.id.1", "item"),
            )
        ],
    }
    g = sess.load_from(edges, vertices)
    return g



if __name__ == '__main__':
    sess = graphscope.session(num_workers=1)
    data_dir = '/root/gstest/ldbc_50'
    #graph = ldbc_50(sess, data_dir)
    graph = load_ldbc50_subgraph(sess, data_dir)
    print('Loaded graph to vineyard')
    print('Get graph learn handle')
    handle = sess.get_gl_handle(graph, 4)
    print('handle created successfully')
    print(handle)
    print("\n\nafter handle\n")
    s = base64.b64decode(handle).decode('utf-8')
    obj = json.loads(s)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\n\n\n\n")
    print(obj)
    print("\n\n\n\n")
    import os
    import sys
    import subprocess
    proc = subprocess.Popen(['python3', '/root/gsa/learning_engine/graph-learn/examples/tf/graphsage/link_prediction.py', handle], stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    # p = os.popen("python3 /root/graph-learn/examples/tf/graphsage/link_prediction.py %s" % handle)
    # print('demo complete!')
    # p.read()

