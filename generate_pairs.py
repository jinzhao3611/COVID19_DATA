from sqlalchemy import create_engine, Column, Integer, TEXT, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://localhost/event_coref', echo=True)
Base = declarative_base(engine)


class Task(Base):
    """
    eg. fields: id, title
    """
    __tablename__ = 'task'
    __table_args__ = {'autoload': True}


class Event(Base):
    """
    eg. fields: id, title
    """
    __tablename__ = 'event'
    __table_args__ = {'autoload': True}

class Article(Base):
    """
    eg. fields: id, title
    """
    __tablename__ = 'article'
    __table_args__ = {'autoload': True}

def loadSession():
    """"""
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


if __name__ == "__main__":
    session = loadSession()
    task_id = 10000
    task_res = session.query(Task).filter(Task.task_id == task_id).first()
    candidates = task_res.candidates.split()

    for query_id in candidates:
        event_res = session.query(Event).filter(Event.id==query_id).first()
        article_id = event_res.article_id
        trigger_id = event_res.trigger_id
        sent_id = int(trigger_id.split("_")[0])
        article_res = session.query(Article).filter(Article.id == article_id).first()
        source = article_res.source
        doc_id = article_res.doc_id

        with open(f"/Users/jinzhao/schoolwork/lab-work/COVID19_DATA/time_pronoun_processed/{source}_{doc_id}.txt", 'r') as f:
            with open(f"/Users/jinzhao/schoolwork/lab-work/COVID19_DATA/elastic_search_output_clusters/{task_id}.txt", 'a') as out_file:
                out_file.write(f.readlines()[sent_id])
                out_file.write('\n')
