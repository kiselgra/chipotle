(require :chipotle)
(in-package :chipotle)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Helper functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(lisp
 (defun LoG (x y sigma)
   (* (- (/ 1 (* (float pi 1.00000) (expt sigma 4))))
      (- 1 (/ (+ (* x x) (* y y))
	      (* 2 sigma sigma)))
      (exp (- (/ (+ (* x x) (* y y))
		 (* 2 sigma sigma))))))

 (defun gaussian (x y sigma)
   (* (/ 1 (* 2 (float pi 1.00000) sigma sigma))
      (exp (- (/ (+ (* x x) (* y y))
		 (* 2 sigma sigma))))))

 (defun box (x y)
   (declare (ignore x y))
   1)

 (defun sample-filter (filter taps &key sigma) ;; no proper integration, just a normalized hack.
   (lisp
    (let ((k (make-array (list taps taps)))
	  (o (floor (/ taps 2)))
	  (sum 0))
      (loop for y from (- o) to o do
	   (loop for x from (- o) to o do
		(let ((val (float (apply filter `(,x ,y ,@(if sigma (list sigma)))))))
		  (setf (aref k (+ y o) (+ x o)) val)
		  (incf sum val))))
      (loop for y from 0 to (1- taps) collect
	   (loop for x from 0 to (1- taps) collect
		(/ (aref k y x) sum))))))
 
 (defun print-filter (kernel)
   (dolist (row kernel)
     (format t "/* [  ")
     (dolist (x row)
       (format t "~,6f  " x))
     (format t "] */~%"))))


(print-filter (sample-filter #'gaussian 5 :sigma 1.25))
(print-filter (sample-filter #'gaussian 7 :sigma 4))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Filter definition
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(chp-preamble)

(filter-graph laplacian

  (edge load-base (:output base) (load-image :file "test.jpg"))
 
  (edge gauss (:input base :output blurred :arch (sse unsigned-char mirror))
    (deflocal
	:mask #.(sample-filter #'gaussian 5 :sigma 1.5)
	:aggregator +
	:operator *))

  (edge laplacian (:input blurred :output lapla :arch (avx unsigned-char repeat))
    (deflocal
	:mask ((-1.0 -1.0 -1.0) 
	       (-1.0  8.0 -1.0)
	       (-1.0 -1.0 -1.0))
	:aggregator +
	:operator *
	:finally (set (lapla 0) (- 255.0f (fabsf r))
		      (lapla 1) (- 255.0f (fabsf g))
		      (lapla 2) (- 255.0f (fabsf b)))
	))

  (edge unblurred-laplacian (:input base :output ublapla :arch (cuda hipacc unsigned-char))
    (deflocal
	:mask ((-1.0 -1.0 -1.0) 
	       (-1.0  8.0 -1.0)
	       (-1.0 -1.0 -1.0))
	:aggregator +
	:operator *
	:finally (set (ublapla 0) (- 255.0f (fabsf r))
		      (ublapla 1) (- 255.0f (fabsf g))
		      (ublapla 2) (- 255.0f (fabsf b)))
	))

  (edge to-grayscale (:input lapla :output gray :arch (cuda unsigned-char))
    (defpoint ()
	(decl ((float r = (lapla 0))
	       (float g = (lapla 1))
	       (float b = (lapla 2))
	       (float luma = (+ (* 0.2126f r) (* 0.7152f g) (* 0.0722f b))))
	  (set (gray 0) luma
	       (gray 1) luma
	       (gray 2) luma))))
 
  (edge gauss2 (:input gray :output output :arch (cuda unsigned-char mirror))
    (deflocal
	:mask #.(sample-filter #'gaussian 7 :sigma 4)
	:aggregator +
	:operator *
	))

  (edge box (:input gray :output boxout :arch (cpu unsigned-char))
    (deflocal
	:mask #.(sample-filter #'box 11)
	:aggregator +
	:operator *
	))

  (edge store-blur (:input blurred)
    (store-image :file "laplace-0-blur.jpg"))

  (edge store-ublapla (:input ublapla)
    (store-image :file "laplace-1-ub.jpg"))
 
  (edge store-lapla (:input lapla)
    (store-image :file "laplace-1.jpg"))

  (edge store-gray (:input gray)
    (store-image :file "laplace-2.jpg"))

  (edge store-out (:input output)
    (store-image :file "laplace-3.jpg"))

  (edge store-boxout (:input boxout)
    (store-image :file "laplace-4.jpg"))
)


