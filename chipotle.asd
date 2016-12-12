(cmu-cuda::cl-reader)
(require :c-mera)
(require :cmu-cuda)
(require :cm-fop)
(cmu-cuda::cm-reader)

(defpackage :chipotle
  (:use :cmu-cuda :cm-fop))

(asdf::defsystem chipotle
  :name "chipotle"
  :version "0.0.1"
  :serial t
  :components ((:file "src/targets")
	       (:file "src/intrinsics")
	       (:file "src/intrinsify")
	       (:file "src/vectorize")
	       (:file "src/chp"))
  :depends-on ("cmu-cuda"
	       "cm-fop"))
