from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, Boolean
import logging
from contextlib import contextmanager


logger = logging.getLogger(__name__)

SQLBase = declarative_base()
Session = sessionmaker()


class EventCannotContinue(Exception):
    pass


@contextmanager
def managed_session():
    session = Session()
    try:
        yield session
        session.commit()
    except EventCannotContinue:
        session.commit()
        raise
    except:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_database(engine):
    SQLBase.metadata.create_all(bind=engine)
    Session.configure(bind=engine)


class ClockOffsets(SQLBase):
    __tablename__ = 'clock_offsets'
    evt_id = Column(Integer, primary_key=True)
    cobo0 = Column(Float)
    cobo1 = Column(Float)
    cobo2 = Column(Float)
    cobo3 = Column(Float)
    cobo4 = Column(Float)
    cobo5 = Column(Float)
    cobo6 = Column(Float)
    cobo7 = Column(Float)
    cobo8 = Column(Float)
    cobo9 = Column(Float)


class ParameterSet(SQLBase):
    __tablename__ = 'params'
    evt_id = Column(Integer, primary_key=True)
    x0 = Column(Float)
    y0 = Column(Float)
    z0 = Column(Float)
    enu0 = Column(Float)
    azi0 = Column(Float)
    pol0 = Column(Float)


class TriggerResult(SQLBase):
    __tablename__ = 'trigger'
    evt_id = Column(Integer, primary_key=True)
    did_trigger = Column(Boolean)
    num_pads_hit = Column(Integer)


class CleaningResult(SQLBase):
    __tablename__ = 'clean'
    evt_id = Column(Integer, primary_key=True)
    num_pts_before = Column(Integer)
    num_pts_after = Column(Integer)


class MinimizerResult(SQLBase):
    __tablename__ = 'minres'
    evt_id = Column(Integer, primary_key=True)
    x0 = Column(Float)
    y0 = Column(Float)
    z0 = Column(Float)
    enu0 = Column(Float)
    azi0 = Column(Float)
    pol0 = Column(Float)
    posChi2 = Column(Float)
    enChi2 = Column(Float)
    vertChi2 = Column(Float)
    lin_scat_ang = Column(Float)
    lin_beam_int = Column(Float)
    lin_chi2 = Column(Float)
    rad_curv = Column(Float)
    brho = Column(Float)
    curv_en = Column(Float)
    curv_ctr_x = Column(Float)
    curv_ctr_y = Column(Float)


def count_finished_events():
    with managed_session() as session:
        num_finished = session.query(ParameterSet).count()
        assert num_finished >= 0, 'Negative number of events finished?'
        return num_finished
