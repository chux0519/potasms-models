"""Original authored personal-preference probes; never organic SMS or private data."""
from collections import Counter
from .normalize import template_text
# Feedback, novel-target and same-topic protected service messages. Parallel
# translations stay grouped. Personal target labels are assigned per user, not
# inferred as universal spam labels from subscription state.
DATA={
'loan':{
'feedback':[
('A lender is advertising a personal loan offer.','贷款机构正在推广个人贷款优惠。'),
('Borrowing rates are promoted in this loan advertisement.','这则贷款广告正在推广借款利率。'),
('Apply for an advertised consumer credit package.','可申请广告中推广的消费信贷套餐。'),
('A finance company is promoting new borrowing options.','金融公司正在推广新的借贷方案。'),
('An advertisement offers installment financing for purchases.','一则广告提供购物分期融资服务。'),
('A credit provider invites applications for its promotional loan.','信贷服务商邀请申请其推广贷款。'),
('Compare the borrowing packages advertised by this lender.','可比较这家机构推广的借款套餐。'),
('A loan agency is marketing a new credit product.','贷款中介正在营销新的信贷产品。'),
('This promotion invites you to borrow for personal spending.','这项推广邀请您为个人消费申请借款。'),
('A consumer lender advertises an application discount.','消费贷款机构宣传申请优惠。')],
'novel':[
('Looking for extra funds? A broker is promoting credit plans.','需要额外资金吗？经纪商正在推广信贷计划。'),
('An advertised cash advance is open to new applicants.','广告中的现金借款面向新申请人开放。'),
('Consider this promotional financing offer for household expenses.','可考虑这项面向家庭开销的融资推广。'),
('The lender marketing team presents flexible borrowing terms.','贷款机构营销团队介绍灵活的借贷条件。'),
('An agent is advertising access to a personal credit line.','代理正在宣传个人授信额度服务。'),
('This campaign promotes a choice of unsecured borrowing plans.','本次活动推广多种无抵押借款方案。')],
'protected':[
('Your existing loan repayment is due tomorrow; check your original account.','您现有贷款的还款日为明天，请查看原账户。'),
('The installment payment for your current loan was received.','您当前贷款的分期还款已收到。'),
('A statement for your active borrowing account is ready.','您正在使用的借款账户账单已生成。'),
('Your requested loan closure confirmation is available in the banking app.','您申请的贷款结清确认可在银行应用中查看。')]},
'jobs':{
'feedback':[
('A recruitment agency advertises open employment positions.','招聘机构正在刊登职位招聘广告。'),
('A hiring campaign invites applicants to browse new jobs.','一项招聘推广邀请求职者浏览新岗位。'),
('A staffing company promotes its vacancy listings.','人力资源公司正在推广其空缺职位目录。'),
('An employment service advertises career opportunities.','就业服务机构正在宣传职业机会。'),
('A recruiter markets a selection of job openings.','招聘人员正在推广一批招聘岗位。'),
('This hiring advertisement seeks new candidates.','这则招聘广告正在寻找新应聘者。'),
('A job agency promotes positions with several employers.','求职中介正在推广多家雇主的岗位。'),
('Browse the vacancies in this recruitment promotion.','可浏览这则招聘推广中的职位空缺。'),
('A talent firm advertises new work opportunities.','人才服务公司正在宣传新的工作机会。'),
('A career platform is marketing its available roles.','职业平台正在推广其可申请岗位。')],
'novel':[
('Employers are seeking applicants through this advertised vacancy campaign.','雇主通过这项职位推广寻找应聘者。'),
('This promotion highlights companies looking for new staff.','这则推广介绍正在寻找新员工的公司。'),
('An agency invites job seekers to explore advertised careers.','中介邀请求职者了解广告中的职业岗位。'),
('A recruitment bulletin promotes several available occupations.','招聘简报正在推广若干可申请职位。'),
('The staffing team advertises work placements for applicants.','招聘团队面向申请人推广工作岗位。'),
('A hiring marketplace is promoting its latest vacancies.','招聘市场正在推广最新的职位空缺。')],
'protected':[
('Your interview for the position you applied to is confirmed for tomorrow.','您已申请职位的面试已确认为明天举行。'),
('Your employer has deposited this month’s salary.','您的雇主已发放本月工资。'),
('The shift change you requested has been approved by your manager.','您申请的调班已获主管批准。'),
('Your accepted job’s onboarding appointment is in your employee account.','您已接受职位的入职预约可在员工账户中查看。')]},
'retail':{
'feedback':[
('A clothing store advertises its seasonal sale.','服装店正在宣传换季促销。'),
('A retailer promotes discounted household products.','零售商正在推广打折家居商品。'),
('This shopping advertisement offers a discount on selected goods.','这则购物广告提供指定商品折扣。'),
('A merchant is marketing a store promotion.','商家正在营销店铺促销活动。'),
('A supermarket advertises special shopping offers.','超市正在宣传购物特价优惠。'),
('The shop promotes savings on its product range.','店铺正在推广商品系列优惠。'),
('A retail campaign invites you to browse sale items.','零售推广邀请您浏览促销商品。'),
('A department store advertises its weekend discounts.','百货商店正在宣传周末折扣。'),
('A seller markets a promotional shopping voucher.','卖家正在推广购物优惠券。'),
('This store advertisement promotes a limited sale event.','这则店铺广告推广限时促销活动。')],
'novel':[
('Selected merchandise is featured in this advertised bargain event.','本次特价推广介绍了部分精选商品。'),
('A merchant’s campaign highlights reduced prices on products.','商家的推广活动介绍了降价商品。'),
('The retailer invites shoppers to explore its promotional catalog.','零售商邀请顾客浏览促销目录。'),
('An advertised shopping event features markdowns across departments.','广告中的购物活动提供多个品类的降价优惠。'),
('A local shop promotes deals for its customer shopping day.','本地店铺为顾客购物日推广优惠。'),
('This commercial message markets a selection of discounted goods.','这条商业信息正在推广一批折扣商品。')],
'protected':[
('The receipt for the purchase you made is ready in your order history.','您已完成购物的收据可在订单历史中查看。'),
('Your requested refund has returned to your original payment account.','您申请的退款已退回原支付账户。'),
('The parcel for your existing order has reached the pickup counter.','您现有订单的包裹已到达取件柜台。'),
('The store confirmed the cancellation you requested for your order.','店铺已确认您申请的订单取消。')]},
'investment':{
'feedback':[
('An investment firm advertises its portfolio products.','投资公司正在宣传其投资组合产品。'),
('A broker promotes an investment account offer.','经纪商正在推广投资账户优惠。'),
('This advertisement markets a new fund subscription.','这则广告正在营销新基金认购。'),
('A wealth service is promoting investment packages.','财富服务机构正在推广投资套餐。'),
('A financial adviser advertises portfolio services.','理财顾问正在宣传投资组合服务。'),
('A brokerage campaign invites new investment clients.','经纪公司的推广活动邀请新的投资客户。'),
('A fund company promotes its available investment products.','基金公司正在推广其投资产品。'),
('An investment advertisement invites you to compare funds.','投资广告邀请您比较基金产品。'),
('A securities firm markets its account opening promotion.','证券公司正在营销开户推广活动。'),
('This promotional notice advertises wealth management options.','这条推广通知宣传财富管理选项。')],
'novel':[
('A campaign introduces advertised opportunities to build a portfolio.','推广活动介绍广告中的投资组合配置机会。'),
('A money manager is marketing access to a selection of funds.','资产管理人员正在推广多种基金服务。'),
('The brokerage advertises services for prospective investors.','经纪公司面向潜在投资者宣传服务。'),
('This commercial bulletin promotes securities investment options.','这则商业简报正在推广证券投资选项。'),
('A financial firm invites customers to explore marketed portfolios.','金融公司邀请客户了解其推广的投资组合。'),
('An adviser promotes a new investment product range.','顾问正在推广新的投资产品系列。')],
'protected':[
('Your existing investment account’s quarterly statement is available.','您现有投资账户的季度对账单已生成。'),
('The securities trade you placed has completed; view the original receipt.','您提交的证券交易已完成，请查看原交易回执。'),
('A dividend from your current holdings was deposited into your account.','您当前持仓的分红已存入您的账户。'),
('Your requested withdrawal from the investment account was processed.','您申请从投资账户提款的操作已处理。')]}}


def build():
    rows=[]
    for topic,roles in DATA.items():
        for role,pairs in roles.items():
            assert len(pairs)=={'feedback':10,'novel':6,'protected':4}[role]
            for i,(en,zh) in enumerate(pairs):
                family=f'learn-{topic}-{role}-{i:02d}'
                for language,text in [('en',en),('zh',zh),('mixed',en+' '+zh.split('，')[0])]:
                    rows.append(dict(id=family+':'+language,family_id=family,template_id=family,topic=topic,role=role,language=language,text=text,
                        label=0,synthetic=True,actual_received_sms=False,personal_preference_target=role!='protected',
                        provenance='original authored fictional notices; aligned EN/ZH; mixed is bilingual composition, not organic code-switch'))
    assert len(rows)==240 and len({r['text'] for r in rows})==240
    for role in ['feedback','novel','protected']:
        for other in ['feedback','novel','protected']:
            if role==other:continue
            assert not {template_text(r['text']) for r in rows if r['role']==role}&{template_text(r['text']) for r in rows if r['role']==other}
    return rows
