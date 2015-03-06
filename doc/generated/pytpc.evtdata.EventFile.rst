pytpc.evtdata.EventFile
=======================

.. currentmodule:: pytpc.evtdata

.. autoclass:: EventFile

   .. rubric:: Working with files

   .. autosummary::
      :toctree:

      ~EventFile.open
      ~EventFile.close
      ~EventFile.make_lookup_table
      ~EventFile.load_lookup_table

   .. rubric:: Reading data

   Most of the commands in this section can be safely ignored. They mainly exist to support the preferred subscripting
   and iteration methods of reading the files.

   .. autosummary::
      :toctree:

      ~EventFile.read_event_by_number
      ~EventFile.read_current
      ~EventFile.read_next
      ~EventFile.read_previous
      ~EventFile.pack_sample
      ~EventFile.unpack_sample
      ~EventFile.unpack_samples
   
   

   
   
   