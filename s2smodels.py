from sqlalchemy import Column,Integer, String,Float,BINARY
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

#create audio_segments tables
class Audio_segment(Base):
    __tablename__ = "audioSegments"
    id = Column(Integer, primary_key=True)
    start_time = Column(Float)
    end_time = Column(Float)
    type = Column(String)
    audio=Column(BINARY)


#create audio_generation table
class AudioGeneration(Base):
    __tablename__ = "audioGeneration"
    id = Column(Integer, primary_key=True)
    audio=Column(BINARY)