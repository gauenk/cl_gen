def deep_eq(_v1, _v2):
  """
  Tests for deep equality between two python data structures recursing 
  into sub-structures if necessary. Works with all python types including
  iterators and generators. This function was dreampt up to test API responses
  but could be used for anything. Be careful. With deeply nested structures
  you may blow the stack.
  
  Doctests included:
  
  >>> x1, y1 = ({'a': 'b'}, {'a': 'b'})
  >>> deep_eq(x1, y1)
  True
  >>> x2, y2 = ({'a': 'b'}, {'b': 'a'})
  >>> deep_eq(x2, y2)
  False
  >>> x3, y3 = ({'a': {'b': 'c'}}, {'a': {'b': 'c'}})
  >>> deep_eq(x3, y3)
  True
  >>> x4, y4 = ({'c': 't', 'a': {'b': 'c'}}, {'a': {'b': 'n'}, 'c': 't'})
  >>> deep_eq(x4, y4)
  False
  >>> x5, y5 = ({'a': [1,2,3]}, {'a': [1,2,3]})
  >>> deep_eq(x5, y5)
  True
  >>> x6, y6 = ({'a': [1,'b',8]}, {'a': [2,'b',8]})
  >>> deep_eq(x6, y6)
  False
  >>> x7, y7 = ('a', 'a')
  >>> deep_eq(x7, y7)
  True
  >>> x8, y8 = (['p','n',['asdf']], ['p','n',['asdf']])
  >>> deep_eq(x8, y8)
  True
  >>> x9, y9 = (['p','n',['asdf',['omg']]], ['p', 'n', ['asdf',['nowai']]])
  >>> deep_eq(x9, y9)
  False
  >>> x10, y10 = (1, 2)
  >>> deep_eq(x10, y10)
  False
  >>> deep_eq((str(p) for p in xrange(10)), (str(p) for p in xrange(10)))
  True
  >>> str(deep_eq(range(4), range(4)))
  'True'  
  >>> deep_eq(xrange(100), xrange(100))
  True
  >>> deep_eq(xrange(2), xrange(5))
  False
  """
  import operator, types
  
  def _deep_dict_eq(d1, d2):
    k1 = sorted(d1.keys())
    k2 = sorted(d2.keys())
    if k1 != k2: # keys should be exactly equal
      return False
    return sum(deep_eq(d1[k], d2[k]) for k in k1) == len(k1)
  
  def _deep_iter_eq(l1, l2):
    if len(l1) != len(l2):
      return False
    return sum(deep_eq(v1, v2) for v1, v2 in zip(l1, l2)) == len(l1)
  
  op = operator.eq
  c1, c2 = (_v1, _v2)
  
  # guard against strings because they are also iterable
  # and will consistently cause a RuntimeError (maximum recursion limit reached)
  for t in [str]:
    if isinstance(_v1, t):
      break
  else:
    if isinstance(_v1, dict):
      op = _deep_dict_eq
    else:
      try:
        c1, c2 = (list(iter(_v1)), list(iter(_v2)))
      except TypeError:
        c1, c2 = _v1, _v2
      else:
        op = _deep_iter_eq
  
  return op(c1, c2)
