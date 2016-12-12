(in-package :chipotle)

;;; ==============================================
;;;                 setup cm-fop
;;; ==============================================

(define-composition-system 3)

;;; ==============================================
;;;                   targets
;;; ==============================================

(define-target default)

(define-target cpu default)
(define-target sse cpu)
(define-target avx sse)

(define-target cuda default)
(define-target hipacc default)

(define-target unsigned-char)
(define-target floating-point)

(define-target point-operator default)

(define-target clamp default)
(define-target exceed default)
(define-target mirror exceed)
(define-target repeat exceed)

;;; ==============================================
;;;       helper functions and sturctures
;;; ==============================================

(defparameter preparations (make-hash-table))
(defparameter prep-done (make-hash-table))
(defparameter store-count (make-hash-table))

;;; from CLHS
(defmacro expand (form &environment env)
  (multiple-value-bind (expansion expanded-p)
      (macroexpand form env)
    `(values ',expansion ',expanded-p)))
(defmacro expand-1 (form &environment env)
  (multiple-value-bind (expansion expanded-p)
      (macroexpand-1 form env)
    `(values ',expansion ',expanded-p)))

(defun ensure-list (x) (cl::if (listp x) x (list x)))

(defmacro hide (&body body)
  `(cl:progn
     ,@body
     (values)))

;;; ==============================================
;;;               generator addons
;;; ==============================================

;;; cgen
(defmacro goto (where)   `(comment ,(format nil "goto ~a;" where) :prefix ""))
(defmacro clabel (which) `(comment ,(format nil "~a:" which) :prefix ""))

;;; cxxgen
(defmacro cerr (&rest args) `(<< cerr ,@args))
(defmacro cout (&rest args) `(<< cout ,@args))
